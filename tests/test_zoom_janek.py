"""ZB1 zoom pilot: CPU tests of tiling, tile-box keeping, coordinate mapping, merge and R3 selection."""
import unittest

from mixab import zoom_janek as zoom 

try:
    import cv2  # noqa: F401
    import shapely  # noqa: F401
    HAVE_GEOM = True
except ImportError:
    HAVE_GEOM = False


def quad(x0, y0, x1, y1):
    return [x0, y0, x1, y0, x1, y1, x0, y1]


class TilingTest(unittest.TestCase):
    def test_cores_partition_and_crops_inside(self):
        for W, H, n in ((512, 384, 2), (513, 385, 3), (1024, 768, 3)):
            tiles = zoom.grid_tiles(W, H, n)
            self.assertEqual(len(tiles), n*n)
            area = sum((t['core'][2] - t['core'][0])*(t['core'][3] - t['core'][1]) for t in tiles)
            self.assertEqual(area, W*H)
            for t in tiles:
                a0, b0, a1, b1 = t['crop']
                x0, y0, x1, y1 = t['core']
                self.assertTrue(0 <= a0 <= x0 and x1 <= a1 <= W and 0 <= b0 <= y0 and y1 <= b1 <= H)

    def test_crop_short_side_is_smaller_so_text_is_enlarged(self):
        # Resize brings the short side to 1024: a crop with a smaller short side is magnified more
        W, H = 512, 384
        t = zoom.grid_tiles(W, H, 2)[0]
        s = min(t['crop'][2] - t['crop'][0], t['crop'][3] - t['crop'][1])
        self.assertAlmostEqual((1024/s)/(1024/H), H/s)
        self.assertGreater(H/s, 1.5)

    def test_keep_tile_box_center_and_edge_rules(self):
        W, H = 512, 384
        tiles = zoom.grid_tiles(W, H, 2)
        t0 = tiles[0]                                    # core [0,0,256,192], crop [0,0,295,221]
        self.assertTrue(zoom.keep_tile_box(quad(10, 10, 60, 20), t0, W, H))
        self.assertFalse(zoom.keep_tile_box(quad(250, 10, 290, 20), t0, W, H))   # center x 270 outside core
        self.assertFalse(zoom.keep_tile_box(quad(200, 10, 294.5, 20), t0, W, H))  # touches interior crop edge 295
        self.assertTrue(zoom.keep_tile_box(quad(0, 0, 30, 10), t0, W, H))         # image border is not a cut
        # every location belongs to exactly one tile
        box = quad(250, 180, 262, 200)                   # center (256, 190) -> tile 1 (core x from 256)
        owners = [t['index'] for t in tiles if zoom.keep_tile_box(box, t, W, H)]
        self.assertEqual(owners, [1])

    def test_coordinate_mapping(self):
        p = zoom.crop_to_task(quad(1, 2, 3, 4), [100, 50, 300, 250])
        self.assertEqual(p, quad(101, 52, 103, 54))
        self.assertEqual(zoom.task_to_clean(p, [2.0, 2.0]), quad(202, 104, 206, 108))


@unittest.skipUnless(HAVE_GEOM, 'needs cv2 + shapely')
class MergeTest(unittest.TestCase):
    def test_merge_keeps_full_adds_small_non_duplicates(self):
        H = 768                                          # Clean height 768 -> LR2 factor 384/768 = .5
        full = [(quad(0, 0, 100, 10), .9)]
        zoom_c = [(quad(0, 0, 100, 10.5), .95, (0, 0)),   # duplicate of the full box (IoU > .5)
                  (quad(200, 0, 260, 10), .8, (0, 1)),    # small (short edge 10 clean px = 5 LR2 px), new -> add
                  (quad(300, 0, 400, 40), .99, (1, 0)),   # large (short edge 20 LR2 px) -> never considered
                  (quad(200, 0, 261, 10), .7, (1, 1))]    # duplicate of the added one -> skip
        kept, added = zoom.merge(full, zoom_c, H)
        self.assertEqual(added, [1])
        self.assertEqual(len(kept), 2)
        kept2, added2 = zoom.merge(full, zoom_c, H, size_cap=float('inf'))
        self.assertEqual(sorted(added2), [1, 2])

    def test_r3_selects_tiles_with_most_small_full_boxes(self):
        W, H = 600, 600
        tiles = zoom.grid_tiles(W, H, 3)
        small = [quad(410 + 5*i, 410, 415 + 5*i, 414) for i in range(3)] + [quad(10, 10, 15, 14)]
        sel = zoom.r3_select(small, H, [1.0, 1.0], tiles, k=2)
        self.assertEqual(sel, [8, 0])


class SampleTest(unittest.TestCase):
    def test_select_sources_split_a_fixed(self):
        from mixab import reselect_split_janek as rs 
        ids = [f'{i:016x}' for i in range(2000)]
        s = zoom.select_sources(ids, 50)
        self.assertEqual(len(s), 50)
        self.assertTrue(all(rs.split_of(x) == 'A' for x in s))
        self.assertEqual(s, zoom.select_sources(list(reversed(ids)), 50))


if __name__ == '__main__':
    unittest.main()
