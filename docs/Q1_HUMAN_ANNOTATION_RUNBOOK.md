# Q1 Human Annotation Runbook

**Áp dụng cho:** `q1-human-gold-package-7aa7c50d4654dd64`  
**Reference ID:** `q1-human-gold-1200-dbc5d7e8fab66bd2`  
**Trạng thái:** Sẵn sàng cho hai người gán blind; chưa có human gold.

## 1. Vai trò và nguyên tắc

- Cần **hai người thật khác nhau**, A và B, cùng gán đủ 1.200 review.
- A/B không trao đổi nhãn, evidence, uncertainty, draft hoặc FINAL trước khi
  cả hai đã khóa.
- Không mở AI-review cho 1.200 sample này trước khi A/B FINAL đều được
  validate/freeze.
- Bộ 200 cũ chỉ là calibration; không dùng thay một trong hai annotator.
- Nếu chỉ có một người gán, output chỉ là single-annotator reference; không
  được báo IAA hoặc double-blind gold trong paper.

## 2. Preflight trước khi bắt đầu

Chạy ở repository root:

```powershell
.\.venv\Scripts\python -X utf8 .\scripts\validate_q1_human_gold_package.py
.\.venv\Scripts\python -X utf8 .\scripts\validate_q1_workbench_validation_release.py
```

Cả hai phải trả `"status": "VALID"`. Không bắt đầu nếu source/package đã
drift.

Package là closed-inventory release. **Không copy draft, FINAL hoặc validated
output vào trong**
`data/annotations/q1_human_gold_1200_v1_20260728/`; thêm file vào đó sẽ làm
package validator fail.

## 3. Chạy annotator A

```powershell
.\human_annotation_ui\start.ps1 -Role A -Port 8765
```

Kiểm tra trên UI:

- banner trên cùng là `HUMAN GOLD · DOUBLE-BLIND`;
- badge là `Annotator A`;
- tiến độ ban đầu là `0 / 1200 hoàn tất`;
- không có banner AI hoặc bảng so sánh A/B.

Nhập mã giả danh ổn định, ví dụ `Q1-ANN-A-01`. Mã bị khóa sau record hoàn tất
đầu tiên.

## 4. Chạy annotator B

Người B dùng máy hoặc browser profile riêng:

```powershell
.\human_annotation_ui\start.ps1 -Role B -Port 8766
```

Kiểm tra badge `Annotator B` và các điều kiện blind giống A. Không dùng cùng
người cho cả A và B.

## 5. Quy trình cho mỗi review

1. Đọc toàn bộ review; tick `Tôi đã đọc toàn bộ review này`.
2. Chọn trạng thái `LABELED`, `ESCALATE` hoặc `REJECT_NON_REVIEW`.
3. Xét độc lập đủ chín aspect. Không dùng cảm xúc chung để lan sang aspect
   không có bằng chứng.
4. Nhãn `-1`, `0`, `1` hoặc `1, -1` phải có exact evidence; mixed phải có
   evidence positive và negative độc lập.
5. Neutral `0` khác absent `2`.
6. Ca mơ hồ phải dùng uncertainty code và `ESCALATE`.
7. Bấm `Kiểm tra`, sau đó `Hoàn tất và sang review kế`.

`Điền các ô trống = Không nhắc` chỉ dùng sau khi đã đọc và xét từng aspect;
action được ghi audit event.

## 6. Lưu tiến độ

- UI autosave theo assignment + workflow hash trong IndexedDB.
- Đóng tab hoặc khóa màn hình không làm mất bản nháp; mở lại đúng browser
  profile/assignment sẽ tiếp tục.
- Sau mỗi 20–25 review, bấm `Sao lưu`; lưu DRAFT vào thư mục riêng ngoài
  frozen package, ví dụ:

  ```text
  C:\ABSA-Q1-SUBMISSIONS\A\drafts\
  C:\ABSA-Q1-SUBMISSIONS\B\drafts\
  ```

- Không gửi draft A cho B hoặc ngược lại.

## 7. Xuất và validate FINAL

Chỉ bấm `Xuất FINAL` khi UI báo 1.200/1.200. Lưu file FINAL ngoài frozen
package.

Validate A:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.validate_export `
  --assignment .\data\annotations\q1_human_gold_1200_v1_20260728\assignments\annotator_a.assignment.json `
  --export C:\ABSA-Q1-SUBMISSIONS\A\<FINAL_A.json> `
  --require-final `
  --output .\data\annotations\q1_human_gold_a_validated_v1_20260728
```

Validate B:

```powershell
.\.venv\Scripts\python -X utf8 -m human_annotation_ui.validate_export `
  --assignment .\data\annotations\q1_human_gold_1200_v1_20260728\assignments\annotator_b.assignment.json `
  --export C:\ABSA-Q1-SUBMISSIONS\B\<FINAL_B.json> `
  --require-final `
  --output .\data\annotations\q1_human_gold_b_validated_v1_20260728
```

Validator phải trả `VALID_FINAL`, `records_completed_valid = 1200`,
`workflow_mode = BLINDED_INDEPENDENT_ANNOTATION`. Output directory là
versioned release mới và validator từ chối overwrite.

## 8. Sau khi cả A và B đã khóa

Thứ tự bắt buộc:

1. Bind/checksum hai validated release.
2. Tính IAA **trước adjudication**: status exact agreement, aspect mention
   F1, per-aspect polarity Cohen's kappa, Krippendorff's alpha và raw
   confusion matrices.
3. Publish IAA report.
4. Build adjudication package chỉ cho disagreement/uncertainty, bind A/B
   export hash và IAA report hash.
5. Expert adjudicate bằng cùng workbench ở `EXPERT_ADJUDICATION` mode.
6. Publish expert-final 1.200 + decision ledger.
7. Group-split core thành dev 300/test 500; giữ challenge 400 báo riêng.

Các bước 2–7 hiện **chưa thực thi** vì chưa có hai human FINAL.

