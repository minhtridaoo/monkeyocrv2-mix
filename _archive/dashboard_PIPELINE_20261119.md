# Pipeline chung: Tuấn + Mix A/B (bản tĩnh để gửi Tuấn)

Bản có thể tick trực tiếp nằm ở `index.html` (cùng thư mục, đọc trạng thái từ `status.js`).

```mermaid
flowchart LR
  subgraph P0["Nền tảng (xong)"]
    T0["✓ T0 B0 · B-Clean · B1"] --> T1["✓ T1 EDA ghép cặp"]
    T0 --> T2["✓ T2 PRC · KD · EDSR"]
    S0["✓ S0 Kế hoạch v3"]
  end
  subgraph P1["Chẩn đoán (Tuấn)"]
    T3["⏳ T3 Bước 3 decision_audit"] --> T4["○ T4 Bước 4 taxonomy"]
    S1["● S1 Chốt 5 điểm"]
  end
  subgraph PA["Mix A-core (bạn)"]
    A1["○ A1 weight_merge + test"] --> A2["○ A2 merge_audit + submit"] --> A3["○ A3 GPU job"]
    A1 --> A4["○ A4 reselect A→B"]
    A3 --> A5["○ A5 đường α + CI · Go/No-Go"]
    A4 --> A5
    A1 -.-> A6["◇ A6 Model Stock"]
  end
  subgraph PB["Mix B (bạn) · song song với A3"]
    B1["○ B1 code FSR + gate-only"] --> B2["○ B2 probe GPU"]
    B2 -.-> B3["◇ B3 train 3 nhánh"] -.-> B4["◇ B4 FSR − gate-only"]
  end
  T1 --> T3
  S0 --> S1 --> A1
  S1 --> B1
  T3 -. "NPZ: kiểm 3b" .-> A5
  T4 -. "điều kiện mở" .-> B3
  A5 -. "điều kiện mở" .-> B3
  A5 --> R1["○ R1 Báo cáo Mix A"] --> R2["○ R2 Gửi Tuấn (24/11)"]
  T4 --> T6["○ T6 Bước 6 Head-control/Residual"]
  R2 --> T8["○ T8 seed 2 + TEST (sau 24/11)"]
```

**Song song hóa:**
- Ngày 21/11, job GPU A3 của Mix A chạy nền.
- 2 giờ công hôm đó dành để code và probe Mix B (B1, B2). Probe không phụ thuộc taxonomy.
- Chỉ bước train B3 bị chặn bởi T4 (taxonomy của Tuấn) và A5 (kết quả Mix A).
- Ký hiệu: `✓` xong · `●` đang làm · `○` chưa làm · `⏳` chờ · `◇` có điều kiện.
