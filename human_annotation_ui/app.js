import {
  ASPECTS,
  ASPECT_UNCERTAINTY_CODES,
  ASSIGNMENT_SCHEMA_VERSION,
  EXPORT_SCHEMA_VERSION,
  UI_VERSION,
  UNCERTAINTY_CODES,
  WORKSPACE_SCHEMA_VERSION,
  canonicalJson,
  codePointLength,
  isRecordUncertain,
  newRecordState,
  validateRecord,
} from "./annotation_core.mjs";

const ASSIGNMENT_KEYS = Object.freeze([
  "schema_version",
  "assignment_id",
  "reference_id",
  "role",
  "item_count",
  "guideline",
  "created_at",
  "records",
  "assignment_payload_sha256",
]);
const ASSIGNMENT_PAYLOAD_KEYS = ASSIGNMENT_KEYS.filter(
  (key) => key !== "assignment_payload_sha256",
);
const RECORD_KEYS = Object.freeze([
  "annotation_id",
  "reviewContent",
  "review_text_sha256",
]);
const GUIDELINE_KEYS = Object.freeze(["document_id", "version", "sha256"]);
const WORKSPACE_KEYS = Object.freeze([
  "schema_version",
  "session_key",
  "assignment_id",
  "assignment_payload_sha256",
  "workflow_id",
  "workflow_mode",
  "workflow_payload_sha256",
  "role",
  "annotator_id",
  "current_index",
  "created_at",
  "updated_at",
  "finalized_at",
  "records",
  "audit_events",
]);
const EXPORT_PAYLOAD_KEYS = Object.freeze([
  "workspace_schema_version",
  "export_status",
  "ui_version",
  "workflow",
  "assignment_id",
  "reference_id",
  "role",
  "assignment_payload_sha256",
  "guideline_version",
  "guideline_sha256",
  "annotator_id",
  "item_count",
  "workspace_created_at",
  "exported_at",
  "finalized_at",
  "records",
  "audit_events",
]);
const WORKSPACE_RECORD_KEYS = Object.freeze([
  "annotation_id",
  "review_text_sha256",
  "annotation_status",
  "aspects",
  "review_uncertainty_codes",
  "notes",
  "read_complete",
  "revisit",
  "complete",
  "completed_at",
  "updated_at",
  "revision_number",
  "revisions",
]);
const ASPECT_KEYS = Object.freeze([
  "aspect",
  "label",
  "evidence",
  "uncertainty_codes",
]);
const EVIDENCE_KEYS = Object.freeze([
  "quote",
  "start",
  "end",
  "polarity",
]);
const REVISION_KEYS = Object.freeze([
  "revision",
  "event",
  "at",
  "annotation",
]);
const ANNOTATION_KEYS = Object.freeze([
  "annotation_status",
  "aspects",
  "review_uncertainty_codes",
  "notes",
]);
const WORKFLOW_ENVELOPE_KEYS = Object.freeze([
  "schema_version",
  "payload",
  "payload_sha256",
]);
const WORKFLOW_PAYLOAD_KEYS = Object.freeze([
  "workflow_id",
  "mode",
  "assignment_id",
  "assignment_payload_sha256",
  "suggestions_available",
  "suggestion_set_id",
  "suggestions_payload_sha256",
  "adjudication_available",
  "adjudication_set_id",
  "adjudication_payload_sha256",
]);
const SUGGESTION_ENVELOPE_KEYS = Object.freeze([
  "schema_version",
  "payload",
  "payload_sha256",
]);
const SUGGESTION_PAYLOAD_KEYS = Object.freeze([
  "suggestion_set_id",
  "assignment_id",
  "reference_id",
  "assignment_payload_sha256",
  "guideline_version",
  "guideline_sha256",
  "item_count",
  "created_at",
  "source_method",
  "records",
]);
const SUGGESTION_RECORD_KEYS = Object.freeze([
  "annotation_id",
  "review_text_sha256",
  ...ANNOTATION_KEYS,
]);
const ADJUDICATION_ENVELOPE_KEYS = Object.freeze([
  "schema_version",
  "payload",
  "payload_sha256",
]);
const ADJUDICATION_PAYLOAD_KEYS = Object.freeze([
  "adjudication_set_id",
  "assignment_id",
  "reference_id",
  "assignment_payload_sha256",
  "guideline_version",
  "guideline_sha256",
  "item_count",
  "created_at",
  "source_a_export_payload_sha256",
  "source_b_export_payload_sha256",
  "source_a_annotator_id",
  "source_b_annotator_id",
  "iaa_report_sha256",
  "records",
]);
const ADJUDICATION_RECORD_KEYS = Object.freeze([
  "annotation_id",
  "review_text_sha256",
  "source_a",
  "source_b",
  "disagreement_fields",
]);
const FORBIDDEN_IMPORT_KEYS = new Set([
  "reviewcontent",
  "rating",
  "category",
  "product_id",
  "seller_id",
  "shop_id",
  "source_url",
  "query",
  "collection_transport",
  "old_labels",
  "llm_labels",
  "model_predictions",
]);
const LABEL_OPTIONS = Object.freeze([
  { value: 2, key: "A", title: "Không nhắc" },
  { value: -1, key: "N", title: "Tiêu cực" },
  { value: 0, key: "U", title: "Trung tính" },
  { value: 1, key: "P", title: "Tích cực" },
  { value: "1, -1", key: "M", title: "Mixed" },
]);
const UNCERTAINTY_LABELS = Object.freeze({
  ASPECT_BOUNDARY: "Ranh giới aspect",
  POLARITY_SCOPE: "Phạm vi polarity",
  SARCASM: "Mỉa mai",
  INSUFFICIENT_CONTEXT: "Thiếu ngữ cảnh",
  TYPO_LANGUAGE: "Lỗi gõ / ngôn ngữ",
  NON_REVIEW: "Không phải review",
  BOILERPLATE_MIXED_WITH_REVIEW: "Boilerplate lẫn review",
  OTHER: "Lý do khác",
});

const $ = (selector) => document.querySelector(selector);
const deepClone = (value) => structuredClone(value);
const nowUtc = () => new Date().toISOString();

let assignment = null;
let workspace = null;
let database = null;
let workflow = null;
let suggestions = null;
let adjudication = null;
let currentIndex = 0;
let focusedAspectIndex = 0;
let savedSelection = null;
let saveTimer = null;
let saveChain = Promise.resolve();
let storageHealthy = true;
let toastTimer = null;
let recordFilter = "all";

function exactKeys(value, keys) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  return (
    actual.length === expected.length &&
    actual.every((key, index) => key === expected[index])
  );
}

function rejectForbiddenKeys(value, path = "$") {
  if (Array.isArray(value)) {
    value.forEach((child, index) =>
      rejectForbiddenKeys(child, `${path}[${index}]`),
    );
    return;
  }
  if (!value || typeof value !== "object") {
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    if (FORBIDDEN_IMPORT_KEYS.has(key.toLowerCase())) {
      throw new Error(`Export chứa field bị cấm tại ${path}.${key}.`);
    }
    rejectForbiddenKeys(child, `${path}.${key}`);
  }
}

async function sha256Text(value) {
  const bytes = new TextEncoder().encode(value);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

async function verifyAssignment(raw) {
  if (!exactKeys(raw, ASSIGNMENT_KEYS)) {
    throw new Error("Assignment có field thiếu, field thừa hoặc metadata bị cấm.");
  }
  if (raw.schema_version !== ASSIGNMENT_SCHEMA_VERSION) {
    throw new Error("Sai phiên bản schema assignment.");
  }
  if (!["A", "B", "REVIEWER", "ADJUDICATOR"].includes(raw.role)) {
    throw new Error(
      "Role assignment phải là A, B, REVIEWER hoặc ADJUDICATOR.",
    );
  }
  if (
    typeof raw.assignment_id !== "string" ||
    !raw.assignment_id ||
    typeof raw.reference_id !== "string" ||
    !raw.reference_id ||
    typeof raw.created_at !== "string"
  ) {
    throw new Error("Định danh assignment không hợp lệ.");
  }
  if (
    !exactKeys(raw.guideline, GUIDELINE_KEYS) ||
    raw.guideline.document_id !== "ABSA-ANNOTATION-GUIDELINE-V2" ||
    raw.guideline.version !== "2.0.0" ||
    !/^[a-f0-9]{64}$/.test(raw.guideline.sha256)
  ) {
    throw new Error("Guideline descriptor không đúng bản V2 đã khóa.");
  }
  if (
    !Number.isInteger(raw.item_count) ||
    raw.item_count < 1 ||
    !Array.isArray(raw.records) ||
    raw.records.length !== raw.item_count
  ) {
    throw new Error("item_count không khớp danh sách review.");
  }
  const ids = new Set();
  const hashes = new Set();
  for (const [index, record] of raw.records.entries()) {
    if (!exactKeys(record, RECORD_KEYS)) {
      throw new Error(
        `Review ${index + 1} có field thiếu/thừa hoặc metadata bị cấm.`,
      );
    }
    if (
      typeof record.annotation_id !== "string" ||
      !record.annotation_id ||
      ids.has(record.annotation_id)
    ) {
      throw new Error(`Review ${index + 1} có annotation ID không hợp lệ.`);
    }
    if (
      typeof record.reviewContent !== "string" ||
      !record.reviewContent ||
      typeof record.review_text_sha256 !== "string"
    ) {
      throw new Error(`Review ${index + 1} có nội dung/hash không hợp lệ.`);
    }
    const actualHash = await sha256Text(record.reviewContent);
    if (actualHash !== record.review_text_sha256) {
      throw new Error(`Review ${index + 1} không khớp SHA-256.`);
    }
    if (hashes.has(actualHash)) {
      throw new Error(`Review ${index + 1} trùng exact text trong assignment.`);
    }
    ids.add(record.annotation_id);
    hashes.add(actualHash);
  }
  const payload = Object.fromEntries(
    ASSIGNMENT_PAYLOAD_KEYS.map((key) => [key, raw[key]]),
  );
  const actualPayloadHash = await sha256Text(canonicalJson(payload));
  if (actualPayloadHash !== raw.assignment_payload_sha256) {
    throw new Error("Checksum toàn assignment không khớp.");
  }
  return raw;
}

async function verifyWorkflow(raw) {
  if (
    !exactKeys(raw, WORKFLOW_ENVELOPE_KEYS) ||
    raw.schema_version !== "human-absa-workflow/2.0.0" ||
    !exactKeys(raw.payload, WORKFLOW_PAYLOAD_KEYS)
  ) {
    throw new Error("Workflow descriptor không hợp lệ.");
  }
  const actualPayloadHash = await sha256Text(canonicalJson(raw.payload));
  if (actualPayloadHash !== raw.payload_sha256) {
    throw new Error("Checksum workflow descriptor không khớp.");
  }
  const payload = raw.payload;
  if (
    payload.assignment_id !== assignment.assignment_id ||
    payload.assignment_payload_sha256 !==
      assignment.assignment_payload_sha256 ||
    typeof payload.workflow_id !== "string" ||
    !payload.workflow_id ||
    ![
      "BLINDED_INDEPENDENT_ANNOTATION",
      "AI_ASSISTED_HUMAN_VERIFICATION",
      "EXPERT_ADJUDICATION",
    ].includes(payload.mode) ||
    typeof payload.suggestions_available !== "boolean" ||
    typeof payload.adjudication_available !== "boolean"
  ) {
    throw new Error("Workflow/assignment descriptor không nhất quán.");
  }
  const aiMode = payload.mode === "AI_ASSISTED_HUMAN_VERIFICATION";
  const adjudicationMode = payload.mode === "EXPERT_ADJUDICATION";
  if (
    aiMode !== payload.suggestions_available ||
    (aiMode &&
      (typeof payload.suggestion_set_id !== "string" ||
        !payload.suggestion_set_id ||
        !/^[a-f0-9]{64}$/.test(payload.suggestions_payload_sha256))) ||
    (!aiMode &&
      (payload.suggestion_set_id !== null ||
        payload.suggestions_payload_sha256 !== null)) ||
    adjudicationMode !== payload.adjudication_available ||
    (adjudicationMode &&
      (typeof payload.adjudication_set_id !== "string" ||
        !payload.adjudication_set_id ||
        !/^[a-f0-9]{64}$/.test(
          payload.adjudication_payload_sha256,
        ))) ||
    (!adjudicationMode &&
      (payload.adjudication_set_id !== null ||
        payload.adjudication_payload_sha256 !== null))
  ) {
    throw new Error("Workflow và auxiliary input không nhất quán.");
  }
  return raw;
}

async function verifySuggestions(raw) {
  if (
    !exactKeys(raw, SUGGESTION_ENVELOPE_KEYS) ||
    raw.schema_version !== "human-absa-ai-suggestions/1.0.0" ||
    !raw.payload ||
    typeof raw.payload !== "object"
  ) {
    throw new Error("AI suggestions không đúng schema.");
  }
  const actualPayloadHash = await sha256Text(canonicalJson(raw.payload));
  if (actualPayloadHash !== raw.payload_sha256) {
    throw new Error("Checksum AI suggestions không khớp.");
  }
  if (
    !exactKeys(raw.payload, SUGGESTION_PAYLOAD_KEYS) ||
    raw.payload.suggestion_set_id !== workflow.payload.suggestion_set_id ||
    raw.payload.assignment_id !== assignment.assignment_id ||
    raw.payload.reference_id !== assignment.reference_id ||
    raw.payload.assignment_payload_sha256 !==
      assignment.assignment_payload_sha256 ||
    raw.payload.guideline_version !== assignment.guideline.version ||
    raw.payload.guideline_sha256 !== assignment.guideline.sha256 ||
    raw.payload.item_count !== assignment.item_count ||
    typeof raw.payload.created_at !== "string" ||
    !raw.payload.created_at ||
    typeof raw.payload.source_method !== "string" ||
    !raw.payload.source_method ||
    !Array.isArray(raw.payload.records) ||
    raw.payload.records.length !== assignment.item_count
  ) {
    throw new Error("AI suggestions không khớp assignment/guideline.");
  }
  rejectForbiddenKeys(raw.payload);
  raw.payload.records.forEach((record, index) => {
    const item = assignment.records[index];
    if (
      !exactKeys(record, SUGGESTION_RECORD_KEYS) ||
      record.annotation_id !== item.annotation_id ||
      record.review_text_sha256 !== item.review_text_sha256
    ) {
      throw new Error(`AI suggestion ${index + 1} sai identity/order.`);
    }
    verifyAnnotationStructure(
      {
        annotation_status: record.annotation_status,
        aspects: record.aspects,
        review_uncertainty_codes: record.review_uncertainty_codes,
        notes: record.notes,
      },
      `AI suggestion ${index + 1}`,
    );
    const candidate = newRecordState(
      record.annotation_id,
      record.review_text_sha256,
    );
    candidate.annotation_status = record.annotation_status;
    candidate.aspects = deepClone(record.aspects);
    candidate.review_uncertainty_codes = deepClone(
      record.review_uncertainty_codes,
    );
    candidate.notes = record.notes;
    candidate.read_complete = true;
    const validation = validateRecord(
      item.reviewContent,
      candidate,
      "AI-SUGGESTION-VALIDATOR",
    );
    if (!validation.valid) {
      throw new Error(
        `AI suggestion ${index + 1} không hợp lệ: ` +
          validation.errors.join(" "),
      );
    }
  });
  return raw;
}

async function verifyAdjudication(raw) {
  if (
    !exactKeys(raw, ADJUDICATION_ENVELOPE_KEYS) ||
    raw.schema_version !== "human-absa-adjudication-input/1.0.0" ||
    !exactKeys(raw.payload, ADJUDICATION_PAYLOAD_KEYS)
  ) {
    throw new Error("Adjudication input không đúng schema.");
  }
  const actualPayloadHash = await sha256Text(canonicalJson(raw.payload));
  if (
    actualPayloadHash !== raw.payload_sha256 ||
    raw.payload_sha256 !==
      workflow.payload.adjudication_payload_sha256
  ) {
    throw new Error("Checksum adjudication input không khớp.");
  }
  if (
    raw.payload.adjudication_set_id !==
      workflow.payload.adjudication_set_id ||
    raw.payload.assignment_id !== assignment.assignment_id ||
    raw.payload.reference_id !== assignment.reference_id ||
    raw.payload.assignment_payload_sha256 !==
      assignment.assignment_payload_sha256 ||
    raw.payload.guideline_version !== assignment.guideline.version ||
    raw.payload.guideline_sha256 !== assignment.guideline.sha256 ||
    raw.payload.item_count !== assignment.item_count ||
    !Array.isArray(raw.payload.records) ||
    raw.payload.records.length !== assignment.item_count
  ) {
    throw new Error("Adjudication input không khớp assignment/guideline.");
  }
  raw.payload.records.forEach((record, index) => {
    const item = assignment.records[index];
    if (
      !exactKeys(record, ADJUDICATION_RECORD_KEYS) ||
      record.annotation_id !== item.annotation_id ||
      record.review_text_sha256 !== item.review_text_sha256 ||
      !Array.isArray(record.disagreement_fields) ||
      record.disagreement_fields.length < 1 ||
      !record.disagreement_fields.every(
        (field) => typeof field === "string" && field,
      )
    ) {
      throw new Error(`Adjudication record ${index + 1} sai identity.`);
    }
    for (const [sourceName, annotation] of [
      ["A", record.source_a],
      ["B", record.source_b],
    ]) {
      verifyAnnotationStructure(
        annotation,
        `Adjudication ${index + 1} source ${sourceName}`,
      );
      const candidate = newRecordState(
        item.annotation_id,
        item.review_text_sha256,
      );
      candidate.annotation_status = annotation.annotation_status;
      candidate.aspects = deepClone(annotation.aspects);
      candidate.review_uncertainty_codes = deepClone(
        annotation.review_uncertainty_codes,
      );
      candidate.notes = annotation.notes;
      candidate.read_complete = true;
      const validation = validateRecord(
        item.reviewContent,
        candidate,
        "ADJUDICATION-SOURCE-VALIDATOR",
      );
      if (!validation.valid) {
        throw new Error(
          `Adjudication ${index + 1} source ${sourceName} invalid: ` +
            validation.errors.join(" "),
        );
      }
    }
    if (
      canonicalJson(record.source_a) === canonicalJson(record.source_b)
    ) {
      throw new Error(`Adjudication record ${index + 1} không có xung đột.`);
    }
  });
  return raw;
}

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("human-absa-annotation-v2", 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains("sessions")) {
        db.createObjectStore("sessions", { keyPath: "session_key" });
      }
    };
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onblocked = () =>
      reject(new Error("IndexedDB bị chặn bởi một tab khác."));
  });
}

function readStoredWorkspace() {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("sessions", "readonly");
    const request = transaction.objectStore("sessions").get(
      `${assignment.assignment_id}:${workflow.payload_sha256}`,
    );
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result || null);
  });
}

function writeStoredWorkspace(snapshot) {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction("sessions", "readwrite");
    transaction.objectStore("sessions").put(snapshot);
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error);
    transaction.onabort = () => reject(transaction.error);
  });
}

function createWorkspace(suggestionEnvelope = null) {
  const records = {};
  const timestamp = nowUtc();
  const suggestionById = new Map(
    (suggestionEnvelope?.payload?.records || []).map((record) => [
      record.annotation_id,
      record,
    ]),
  );
  for (const item of assignment.records) {
    const state = newRecordState(
      item.annotation_id,
      item.review_text_sha256,
    );
    const suggestion = suggestionById.get(item.annotation_id);
    if (suggestion) {
      state.annotation_status = suggestion.annotation_status;
      state.aspects = deepClone(suggestion.aspects);
      state.review_uncertainty_codes = deepClone(
        suggestion.review_uncertainty_codes,
      );
      state.notes = suggestion.notes;
      state.updated_at = timestamp;
    }
    records[item.annotation_id] = state;
  }
  const auditEvents = [
    {
      event: "WORKSPACE_CREATED",
      workflow_id: workflow.payload.workflow_id,
      workflow_mode: workflow.payload.mode,
      workflow_payload_sha256: workflow.payload_sha256,
      at: timestamp,
    },
  ];
  if (suggestionEnvelope) {
    auditEvents.push({
      event: "AI_SUGGESTIONS_SEEDED_FOR_HUMAN_VERIFICATION",
      at: timestamp,
      suggestion_set_id:
        suggestionEnvelope.payload.suggestion_set_id,
      suggestions_payload_sha256:
        suggestionEnvelope.payload_sha256,
    });
  }
  return {
    schema_version: WORKSPACE_SCHEMA_VERSION,
    session_key:
      `${assignment.assignment_id}:${workflow.payload_sha256}`,
    assignment_id: assignment.assignment_id,
    assignment_payload_sha256: assignment.assignment_payload_sha256,
    workflow_id: workflow.payload.workflow_id,
    workflow_mode: workflow.payload.mode,
    workflow_payload_sha256: workflow.payload_sha256,
    role: assignment.role,
    annotator_id: "",
    current_index: 0,
    created_at: timestamp,
    updated_at: timestamp,
    finalized_at: null,
    records,
    audit_events: auditEvents,
  };
}

function verifyStoredWorkspace(value) {
  if (
    !value ||
    !exactKeys(value, WORKSPACE_KEYS) ||
    value.schema_version !== WORKSPACE_SCHEMA_VERSION ||
    value.session_key !==
      `${assignment.assignment_id}:${workflow.payload_sha256}` ||
    value.assignment_id !== assignment.assignment_id ||
    value.assignment_payload_sha256 !==
      assignment.assignment_payload_sha256 ||
    value.workflow_id !== workflow.payload.workflow_id ||
    value.workflow_mode !== workflow.payload.mode ||
    value.workflow_payload_sha256 !== workflow.payload_sha256 ||
    value.role !== assignment.role ||
    !value.records ||
    typeof value.records !== "object"
  ) {
    throw new Error(
      "Bản nháp trong browser không khớp assignment hiện tại. Không tự động ghi đè.",
    );
  }
  const expectedIds = new Set(
    assignment.records.map((item) => item.annotation_id),
  );
  const storedIds = new Set(Object.keys(value.records));
  if (
    expectedIds.size !== storedIds.size ||
    [...expectedIds].some((id) => !storedIds.has(id))
  ) {
    throw new Error("Bản nháp trong browser có tập review không hợp lệ.");
  }
  for (const item of assignment.records) {
    verifyWorkspaceRecordStructure(
      value.records[item.annotation_id],
      item,
      `Bản nháp ${item.annotation_id}`,
    );
  }
  if (
    !Array.isArray(value.audit_events) ||
    typeof value.annotator_id !== "string"
  ) {
    throw new Error("Bản nháp trong browser có audit/annotator sai schema.");
  }
  return value;
}

function verifyAnnotationStructure(annotation, context) {
  if (!exactKeys(annotation, ANNOTATION_KEYS)) {
    throw new Error(`${context} có field annotation thiếu/thừa.`);
  }
  if (
    !Array.isArray(annotation.aspects) ||
    annotation.aspects.length !== ASPECTS.length ||
    !Array.isArray(annotation.review_uncertainty_codes) ||
    typeof annotation.notes !== "string"
  ) {
    throw new Error(`${context} có cấu trúc annotation không hợp lệ.`);
  }
  annotation.aspects.forEach((aspect, index) => {
    if (
      !exactKeys(aspect, ASPECT_KEYS) ||
      aspect.aspect !== ASPECTS[index] ||
      !Array.isArray(aspect.evidence) ||
      !Array.isArray(aspect.uncertainty_codes)
    ) {
      throw new Error(`${context} sai aspect ${index + 1}.`);
    }
    aspect.evidence.forEach((evidence, evidenceIndex) => {
      if (!exactKeys(evidence, EVIDENCE_KEYS)) {
        throw new Error(
          `${context} sai evidence ${index + 1}.${evidenceIndex + 1}.`,
        );
      }
    });
  });
}

function verifyWorkspaceRecordStructure(record, item, context) {
  if (!exactKeys(record, WORKSPACE_RECORD_KEYS)) {
    throw new Error(`${context} có field record thiếu/thừa.`);
  }
  if (
    record.annotation_id !== item.annotation_id ||
    record.review_text_sha256 !== item.review_text_sha256 ||
    typeof record.read_complete !== "boolean" ||
    typeof record.revisit !== "boolean" ||
    typeof record.complete !== "boolean" ||
    !Number.isInteger(record.revision_number) ||
    record.revision_number < 0 ||
    !Array.isArray(record.revisions) ||
    record.revisions.length !== record.revision_number
  ) {
    throw new Error(`${context} có identity/state/revision không hợp lệ.`);
  }
  verifyAnnotationStructure(
    {
      annotation_status: record.annotation_status,
      aspects: record.aspects,
      review_uncertainty_codes: record.review_uncertainty_codes,
      notes: record.notes,
    },
    context,
  );
  record.revisions.forEach((revision, index) => {
    if (
      !exactKeys(revision, REVISION_KEYS) ||
      revision.revision !== index + 1 ||
      revision.event !== "COMPLETED" ||
      typeof revision.at !== "string"
    ) {
      throw new Error(`${context} có revision ${index + 1} không hợp lệ.`);
    }
    verifyAnnotationStructure(
      revision.annotation,
      `${context} revision ${index + 1}`,
    );
  });
}

async function persistState() {
  if (!workspace) {
    return;
  }
  $("#saveState").textContent = "Đang lưu…";
  $("#saveState").classList.remove("failed");
  try {
    workspace.updated_at = nowUtc();
    await writeStoredWorkspace(deepClone(workspace));
    storageHealthy = true;
    $("#saveState").textContent = "Đã lưu";
  } catch (error) {
    storageHealthy = false;
    $("#saveState").textContent = "Lưu thất bại";
    $("#saveState").classList.add("failed");
    showToast(
      `Không thể lưu nháp: ${error?.message || "lỗi storage"}. Không hoàn tất review cho đến khi lưu lại được.`,
      9000,
    );
    throw error;
  }
}

function queueSave() {
  $("#saveState").textContent = "Chờ lưu…";
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
  }
  saveTimer = setTimeout(() => {
    saveTimer = null;
    saveChain = saveChain.then(persistState, persistState).catch(() => {});
  }, 260);
}

async function flushSave() {
  if (saveTimer !== null) {
    clearTimeout(saveTimer);
    saveTimer = null;
  }
  saveChain = saveChain.then(persistState, persistState);
  return saveChain;
}

function showToast(message, duration = 3500) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  if (toastTimer !== null) {
    clearTimeout(toastTimer);
  }
  toastTimer = setTimeout(() => {
    toast.hidden = true;
  }, duration);
}

function showFatal(title, message) {
  $("#bootScreen").hidden = true;
  $("#appShell").hidden = true;
  $("#fatalTitle").textContent = title;
  $("#fatalMessage").textContent = message;
  $("#fatalScreen").hidden = false;
}

function currentAssignmentRecord() {
  return assignment.records[currentIndex];
}

function currentRecordState() {
  return workspace.records[currentAssignmentRecord().annotation_id];
}

function currentAdjudicationRecord() {
  if (!adjudication) {
    return null;
  }
  return adjudication.payload.records[currentIndex];
}

function countCompleted() {
  return Object.values(workspace.records).filter((record) => record.complete)
    .length;
}

function reopenIfCompleted(record, reason) {
  if (!record.complete) {
    return;
  }
  record.complete = false;
  record.completed_at = null;
  workspace.audit_events.push({
    event: "RECORD_REOPENED",
    annotation_id: record.annotation_id,
    reason,
    at: nowUtc(),
  });
}

function markRecordChanged(reason, { rerender = true } = {}) {
  const record = currentRecordState();
  reopenIfCompleted(record, reason);
  record.updated_at = nowUtc();
  workspace.updated_at = record.updated_at;
  queueSave();
  refreshHeader();
  if (rerender) {
    renderRecord();
  } else {
    renderRecordState();
  }
}

function renderHeaderIdentity() {
  const identity = {
    BLINDED_INDEPENDENT_ANNOTATION: `Annotator ${assignment.role}`,
    AI_ASSISTED_HUMAN_VERIFICATION: "Human reviewer",
    EXPERT_ADJUDICATION: "Expert adjudicator",
  }[workflow.payload.mode];
  $("#roleBadge").textContent = identity;
  $("#assignmentLabel").textContent =
    `Assignment ${assignment.assignment_id.slice(0, 18)}…`;
  $("#recordsDialogTitle").textContent =
    `${assignment.item_count} review`;
  const progressBar = $(".progress-track");
  progressBar.setAttribute("aria-valuemax", String(assignment.item_count));
}

function refreshHeader() {
  const complete = countCompleted();
  const total = assignment.item_count;
  $("#progressText").textContent = `${complete} / ${total} hoàn tất`;
  $("#progressFill").style.width = `${(complete / total) * 100}%`;
  $(".progress-track").setAttribute("aria-valuenow", String(complete));
  $("#annotatorIdInput").disabled =
    Boolean(workspace.finalized_at) || complete > 0;
  renderRecordState();
}

function renderRecordState() {
  const record = currentRecordState();
  const badge = $("#recordStateBadge");
  badge.className = "record-state";
  if (!record.complete) {
    badge.textContent = "Chưa hoàn tất";
    badge.classList.add("incomplete");
  } else if (record.annotation_status === "ESCALATE") {
    badge.textContent = "Đã khóa · cần hòa giải";
    badge.classList.add("escalate");
  } else {
    badge.textContent = "Đã hoàn tất";
  }
  $("#revisitButton").setAttribute(
    "aria-pressed",
    String(Boolean(record.revisit)),
  );
  $("#revisitButton").textContent = record.revisit
    ? "Đã đánh dấu xem lại"
    : "Đánh dấu xem lại";
}

function renderRecord() {
  savedSelection = null;
  $("#selectionTray").hidden = true;
  const item = currentAssignmentRecord();
  const record = currentRecordState();
  $("#recordPosition").textContent =
    `${currentIndex + 1} / ${assignment.item_count}`;
  $("#annotationId").textContent = item.annotation_id;
  $("#reviewText").textContent = item.reviewContent;
  renderAdjudication();
  $("#readCompleteCheckbox").checked = Boolean(record.read_complete);
  $("#notesInput").value = record.notes;
  $("#previousButton").disabled = currentIndex === 0;
  $("#nextButton").disabled = currentIndex === assignment.item_count - 1;
  for (const input of document.querySelectorAll(
    'input[name="annotationStatus"]',
  )) {
    input.checked = input.value === record.annotation_status;
    input.disabled = Boolean(workspace.finalized_at);
  }
  renderAspects(record);
  renderReviewUncertainty(record);
  $("#readCompleteCheckbox").disabled = Boolean(workspace.finalized_at);
  $("#notesInput").disabled = Boolean(workspace.finalized_at);
  $("#fillAbsentButton").disabled =
    Boolean(workspace.finalized_at) || !record.read_complete;
  $("#completeButton").disabled = Boolean(workspace.finalized_at);
  $("#validateButton").disabled = Boolean(workspace.finalized_at);
  $("#completeButton").textContent = record.complete
    ? "Đã hoàn tất"
    : "Hoàn tất và sang review kế";
  $("#validationPanel").hidden = true;
  renderRecordState();
}

function annotationResolution(record) {
  const comparison = currentAdjudicationRecord();
  if (!comparison) {
    return null;
  }
  const current = canonicalJson(exportAnnotationSnapshot(record));
  const sourceA = canonicalJson(comparison.source_a);
  const sourceB = canonicalJson(comparison.source_b);
  if (current === sourceA && current === sourceB) {
    return "MATCHES_BOTH";
  }
  if (current === sourceA) {
    return "ACCEPT_A";
  }
  if (current === sourceB) {
    return "ACCEPT_B";
  }
  return "MANUAL_OVERRIDE";
}

function annotationSummary(container, annotation) {
  const list = document.createElement("dl");
  list.className = "comparison-summary";
  const addRow = (term, value, evidence = []) => {
    const row = document.createElement("div");
    row.className = "comparison-row";
    const key = document.createElement("dt");
    key.textContent = term;
    const description = document.createElement("dd");
    description.textContent = value;
    if (evidence.length > 0) {
      const detail = document.createElement("span");
      detail.className = "comparison-evidence";
      detail.textContent = evidence
        .map((item) => `“${item.quote}” (${item.polarity})`)
        .join(" · ");
      description.append(detail);
    }
    row.append(key, description);
    list.append(row);
  };
  addRow("Trạng thái", annotation.annotation_status);
  for (const aspect of annotation.aspects) {
    if (aspect.label === 2 || aspect.label === null) {
      continue;
    }
    const option = LABEL_OPTIONS.find(
      (candidate) => candidate.value === aspect.label,
    );
    addRow(
      aspect.aspect,
      `${String(aspect.label)} · ${option?.title || "Nhãn"}`,
      aspect.evidence,
    );
  }
  if (annotation.review_uncertainty_codes.length > 0) {
    addRow(
      "Uncertainty",
      annotation.review_uncertainty_codes.join(", "),
    );
  }
  if (annotation.notes) {
    addRow("Ghi chú nguồn", annotation.notes);
  }
  container.replaceChildren(list);
}

function renderAdjudication() {
  const panel = $("#adjudicationPanel");
  const comparison = currentAdjudicationRecord();
  if (!comparison) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $("#disagreementCount").textContent =
    `${comparison.disagreement_fields.length} khác biệt`;
  $("#disagreementFields").textContent =
    `Trường cần adjudicate: ${comparison.disagreement_fields.join(", ")}`;
  annotationSummary($("#adjudicationSourceA"), comparison.source_a);
  annotationSummary($("#adjudicationSourceB"), comparison.source_b);
  $("#applySourceAButton").disabled = Boolean(workspace.finalized_at);
  $("#applySourceBButton").disabled = Boolean(workspace.finalized_at);
}

function applyAdjudicationSource(sourceName) {
  if (workspace.finalized_at || !adjudication) {
    return;
  }
  const comparison = currentAdjudicationRecord();
  const source =
    sourceName === "A" ? comparison.source_a : comparison.source_b;
  const record = currentRecordState();
  reopenIfCompleted(record, `ADJUDICATION_SOURCE_${sourceName}_APPLIED`);
  record.annotation_status = source.annotation_status;
  record.aspects = deepClone(source.aspects);
  record.review_uncertainty_codes = deepClone(
    source.review_uncertainty_codes,
  );
  record.notes = source.notes;
  record.updated_at = nowUtc();
  workspace.audit_events.push({
    event: "ADJUDICATION_SOURCE_APPLIED",
    annotation_id: record.annotation_id,
    source: sourceName,
    at: record.updated_at,
  });
  queueSave();
  renderRecord();
  refreshHeader();
}

function renderAspects(record) {
  const list = $("#aspectList");
  list.replaceChildren();
  const reject = record.annotation_status === "REJECT_NON_REVIEW";
  ASPECTS.forEach((aspect, index) => {
    const rowState = record.aspects[index];
    const row = document.createElement("section");
    row.className = "aspect-row";
    row.dataset.aspectIndex = String(index);
    if (index === focusedAspectIndex) {
      row.classList.add("focused");
    }
    row.addEventListener("focusin", () => {
      focusedAspectIndex = index;
      document
        .querySelectorAll(".aspect-row")
        .forEach((node) => node.classList.remove("focused"));
      row.classList.add("focused");
    });

    const top = document.createElement("div");
    top.className = "aspect-topline";
    const title = document.createElement("div");
    title.className = "aspect-title";
    const number = document.createElement("span");
    number.className = "aspect-index";
    number.textContent = String(index + 1);
    const heading = document.createElement("h3");
    heading.textContent = aspect;
    title.append(number, heading);
    top.append(title);
    row.append(top);

    const labels = document.createElement("div");
    labels.className = "label-options";
    labels.setAttribute("role", "radiogroup");
    labels.setAttribute("aria-label", `Nhãn cho ${aspect}`);
    for (const option of LABEL_OPTIONS) {
      const label = document.createElement("label");
      label.className = "label-option";
      label.dataset.label = String(option.value);
      const input = document.createElement("input");
      input.type = "radio";
      input.name = `aspect-${index}`;
      input.value = String(option.value);
      input.checked = rowState.label === option.value;
      input.disabled = reject || Boolean(workspace.finalized_at);
      input.addEventListener("change", async () => {
        if (!input.checked) {
          return;
        }
        await setAspectLabel(index, option.value);
      });
      const labelText = document.createElement("span");
      const valueText = document.createElement("b");
      valueText.textContent = String(option.value);
      labelText.append(valueText, document.createTextNode(option.title));
      label.append(input, labelText);
      labels.append(label);
    }
    row.append(labels);

    const evidenceArea = document.createElement("div");
    evidenceArea.className = "evidence-area";
    const evidenceList = document.createElement("div");
    evidenceList.className = "evidence-list";
    if (rowState.evidence.length === 0) {
      const empty = document.createElement("span");
      empty.className = "evidence-empty";
      empty.textContent =
        rowState.label === 2 || reject
          ? "Không cần evidence"
          : "Chưa có evidence";
      evidenceList.append(empty);
    } else {
      rowState.evidence.forEach((item, evidenceIndex) => {
        const chip = document.createElement("span");
        chip.className = "evidence-chip";
        const polarity = document.createElement("span");
        polarity.className = "evidence-polarity";
        polarity.textContent =
          item.polarity === "positive"
            ? "POS"
            : item.polarity === "negative"
              ? "NEG"
              : "NEU";
        const quote = document.createElement("q");
        quote.textContent = item.quote;
        quote.title = `${item.start}:${item.end}`;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "remove-evidence";
        remove.textContent = "×";
        remove.title = "Xóa evidence này";
        remove.setAttribute(
          "aria-label",
          `Xóa evidence ${evidenceIndex + 1} của ${aspect}`,
        );
        remove.disabled = Boolean(workspace.finalized_at);
        remove.addEventListener("click", () => {
          rowState.evidence.splice(evidenceIndex, 1);
          markRecordChanged("EVIDENCE_REMOVED");
        });
        chip.append(polarity, quote, remove);
        evidenceList.append(chip);
      });
    }
    evidenceArea.append(evidenceList);

    const evidenceActions = document.createElement("div");
    evidenceActions.className = "evidence-actions";
    if (!reject && rowState.label !== null && rowState.label !== 2) {
      if (rowState.label === "1, -1") {
        evidenceActions.append(
          createEvidenceButton(index, "positive", "+ Positive"),
          createEvidenceButton(index, "negative", "+ Negative"),
        );
      } else {
        const polarity =
          rowState.label === 1
            ? "positive"
            : rowState.label === -1
              ? "negative"
              : "neutral";
        evidenceActions.append(
          createEvidenceButton(index, polarity, "+ Evidence"),
        );
      }
    }
    evidenceArea.append(evidenceActions);
    row.append(evidenceArea);

    const uncertainty = document.createElement("details");
    uncertainty.className = "aspect-uncertainty";
    const summary = document.createElement("summary");
    summary.textContent = rowState.uncertainty_codes.length
      ? `Không chắc · ${rowState.uncertainty_codes.length} mã`
      : "Đánh dấu không chắc";
    uncertainty.append(summary);
    const checks = document.createElement("div");
    checks.className = "checkbox-grid";
    for (const code of ASPECT_UNCERTAINTY_CODES) {
      checks.append(
        createUncertaintyCheckbox(
          code,
          rowState.uncertainty_codes,
          (checked) => {
            toggleCode(rowState.uncertainty_codes, code, checked);
            ensureEscalateForUncertainty();
            markRecordChanged("ASPECT_UNCERTAINTY_CHANGED");
          },
          Boolean(workspace.finalized_at) || reject,
        ),
      );
    }
    uncertainty.append(checks);
    row.append(uncertainty);
    list.append(row);
  });
}

function createEvidenceButton(index, polarity, title) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "evidence-button";
  button.textContent = title;
  button.disabled = Boolean(workspace.finalized_at);
  button.addEventListener("click", () => addEvidence(index, polarity));
  return button;
}

function createUncertaintyCheckbox(code, selected, onChange, disabled) {
  const label = document.createElement("label");
  const input = document.createElement("input");
  input.type = "checkbox";
  input.value = code;
  input.checked = selected.includes(code);
  input.disabled = disabled;
  input.addEventListener("change", () => onChange(input.checked));
  label.append(input, document.createTextNode(UNCERTAINTY_LABELS[code]));
  return label;
}

function renderReviewUncertainty(record) {
  const container = $("#reviewUncertainty");
  container.replaceChildren();
  for (const code of UNCERTAINTY_CODES) {
    container.append(
      createUncertaintyCheckbox(
        code,
        record.review_uncertainty_codes,
        (checked) => {
          if (
            code === "NON_REVIEW" &&
            checked &&
            record.annotation_status !== "REJECT_NON_REVIEW"
          ) {
            showToast(
              "NON_REVIEW chỉ dùng khi chọn trạng thái Không phải review.",
            );
            renderRecord();
            return;
          }
          toggleCode(record.review_uncertainty_codes, code, checked);
          ensureEscalateForUncertainty();
          markRecordChanged("REVIEW_UNCERTAINTY_CHANGED");
        },
        Boolean(workspace.finalized_at),
      ),
    );
  }
}

function toggleCode(target, code, checked) {
  const index = target.indexOf(code);
  if (checked && index < 0) {
    target.push(code);
  } else if (!checked && index >= 0) {
    target.splice(index, 1);
  }
}

function ensureEscalateForUncertainty() {
  const record = currentRecordState();
  const hasUncertainty =
    record.review_uncertainty_codes.length > 0 ||
    record.aspects.some((row) => row.uncertainty_codes.length > 0);
  if (
    hasUncertainty &&
    record.annotation_status !== "REJECT_NON_REVIEW" &&
    record.annotation_status !== "ESCALATE"
  ) {
    record.annotation_status = "ESCALATE";
    showToast("Đã chuyển trạng thái sang Cần hòa giải vì có uncertainty.");
  }
}

async function setAspectLabel(index, value) {
  const record = currentRecordState();
  const row = record.aspects[index];
  if (value === 2 && row.evidence.length > 0) {
    const accepted = await askConfirm({
      title: "Xóa evidence của aspect này?",
      message:
        "Nhãn 2 nghĩa là aspect hoàn toàn không được nhắc nên evidence hiện có sẽ bị xóa.",
      acceptText: "Đổi sang 2 và xóa",
    });
    if (!accepted) {
      renderRecord();
      return;
    }
    row.evidence = [];
  }
  row.label = value;
  focusedAspectIndex = index;
  markRecordChanged("ASPECT_LABEL_CHANGED");
}

function captureReviewSelection() {
  const selection = window.getSelection();
  const reviewNode = $("#reviewText");
  if (!selection || selection.rangeCount !== 1 || selection.isCollapsed) {
    return;
  }
  const range = selection.getRangeAt(0);
  if (
    !reviewNode.contains(range.startContainer) ||
    !reviewNode.contains(range.endContainer)
  ) {
    return;
  }
  const prefixRange = document.createRange();
  prefixRange.selectNodeContents(reviewNode);
  prefixRange.setEnd(range.startContainer, range.startOffset);
  const start = codePointLength(prefixRange.toString());
  const selectedText = range.toString();
  const end = start + codePointLength(selectedText);
  const review = currentAssignmentRecord().reviewContent;
  const exact = Array.from(review).slice(start, end).join("");
  if (!exact || exact !== selectedText) {
    showToast("Không thể ánh xạ đoạn chọn vào nguyên văn review.");
    return;
  }
  savedSelection = {
    annotation_id: currentAssignmentRecord().annotation_id,
    quote: exact,
    start,
    end,
  };
  $("#selectionPreview").textContent = exact;
  $("#selectionTray").hidden = false;
}

function clearSelection() {
  savedSelection = null;
  window.getSelection()?.removeAllRanges();
  $("#selectionTray").hidden = true;
}

function addEvidence(index, polarity) {
  const record = currentRecordState();
  const row = record.aspects[index];
  focusedAspectIndex = index;
  if (
    !savedSelection ||
    savedSelection.annotation_id !== currentAssignmentRecord().annotation_id
  ) {
    showToast("Hãy bôi đen exact text trong review trước.");
    $("#reviewText").focus();
    return;
  }
  if (!savedSelection.quote.trim()) {
    showToast("Evidence không được chỉ chứa khoảng trắng.");
    return;
  }
  if (row.label === null || row.label === 2) {
    showToast("Hãy chọn một nhãn mentioned trước khi thêm evidence.");
    return;
  }
  if (
    row.evidence.some(
      (item) =>
        item.start === savedSelection.start && item.end === savedSelection.end,
    )
  ) {
    showToast("Occurrence này đã được dùng cho aspect hiện tại.");
    return;
  }
  row.evidence.push({
    quote: savedSelection.quote,
    start: savedSelection.start,
    end: savedSelection.end,
    polarity,
  });
  clearSelection();
  markRecordChanged("EVIDENCE_ADDED");
}

function showValidation(result) {
  const panel = $("#validationPanel");
  const list = $("#validationList");
  list.replaceChildren();
  if (result.valid) {
    panel.hidden = true;
    showToast("Review hợp lệ theo Guideline V2.");
    return;
  }
  for (const error of result.errors) {
    const item = document.createElement("li");
    item.textContent = error;
    list.append(item);
  }
  panel.hidden = false;
  panel.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function validateCurrent() {
  const result = validateRecord(
    currentAssignmentRecord().reviewContent,
    currentRecordState(),
    workspace.annotator_id,
  );
  if (
    result.valid &&
    workflow.payload.mode === "EXPERT_ADJUDICATION" &&
    annotationResolution(currentRecordState()) === "MANUAL_OVERRIDE"
  ) {
    const comparison = currentAdjudicationRecord();
    const notes = currentRecordState().notes.trim();
    if (
      !notes ||
      notes === comparison.source_a.notes.trim() ||
      notes === comparison.source_b.notes.trim()
    ) {
      result.valid = false;
      result.errors.push(
        "Manual override phải có lý do expert mới trong ô Ghi chú; " +
          "không được giữ nguyên ghi chú nguồn A/B.",
      );
    }
  }
  showValidation(result);
  return result;
}

async function completeCurrent() {
  if (workspace.finalized_at) {
    return;
  }
  const result = validateCurrent();
  if (!result.valid) {
    return;
  }
  const record = currentRecordState();
  const backup = deepClone(record);
  const previousIndex = currentIndex;
  const timestamp = nowUtc();
  record.complete = true;
  record.completed_at = timestamp;
  record.updated_at = timestamp;
  record.revision_number += 1;
  record.revisions.push({
    revision: record.revision_number,
    event: "COMPLETED",
    at: timestamp,
    annotation: exportAnnotationSnapshot(record),
  });
  workspace.audit_events.push({
    event: "RECORD_COMPLETED",
    annotation_id: record.annotation_id,
    revision: record.revision_number,
    at: timestamp,
  });
  if (workflow.payload.mode === "EXPERT_ADJUDICATION") {
    workspace.audit_events.push({
      event: "ADJUDICATION_DECISION",
      annotation_id: record.annotation_id,
      revision: record.revision_number,
      resolution_basis: annotationResolution(record),
      disagreement_fields: [
        ...currentAdjudicationRecord().disagreement_fields,
      ],
      at: timestamp,
    });
  }
  const next = findNextIncomplete(currentIndex);
  if (next !== null) {
    currentIndex = next;
    workspace.current_index = next;
  }
  try {
    await flushSave();
  } catch {
    workspace.records[record.annotation_id] = backup;
    currentIndex = previousIndex;
    workspace.current_index = previousIndex;
    renderRecord();
    refreshHeader();
    return;
  }
  storageHealthy = true;
  refreshHeader();
  if (next === null) {
    renderRecord();
    showToast("Đã hoàn tất toàn bộ assignment. Có thể xuất FINAL.", 7000);
  } else {
    renderRecord();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }
}

function exportAnnotationSnapshot(record) {
  return {
    annotation_status: record.annotation_status,
    aspects: record.aspects.map((row) => ({
      aspect: row.aspect,
      label: row.label,
      evidence: row.evidence.map((item) => ({ ...item })),
      uncertainty_codes: [...row.uncertainty_codes],
    })),
    review_uncertainty_codes: [...record.review_uncertainty_codes],
    notes: record.notes,
  };
}

function findNextIncomplete(startIndex) {
  for (let offset = 1; offset <= assignment.item_count; offset += 1) {
    const candidate = (startIndex + offset) % assignment.item_count;
    const item = assignment.records[candidate];
    if (!workspace.records[item.annotation_id].complete) {
      return candidate;
    }
  }
  return null;
}

function navigateTo(index) {
  if (index < 0 || index >= assignment.item_count) {
    return;
  }
  currentIndex = index;
  workspace.current_index = index;
  queueSave();
  renderRecord();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderRecordGrid() {
  const grid = $("#recordGrid");
  grid.replaceChildren();
  assignment.records.forEach((item, index) => {
    const record = workspace.records[item.annotation_id];
    const uncertain = isRecordUncertain(record);
    const visible =
      recordFilter === "all" ||
      (recordFilter === "complete" && record.complete) ||
      (recordFilter === "incomplete" && !record.complete) ||
      (recordFilter === "revisit" && record.revisit) ||
      (recordFilter === "uncertain" && uncertain);
    if (!visible) {
      return;
    }
    const button = document.createElement("button");
    button.type = "button";
    button.className = "record-cell";
    button.textContent = String(index + 1);
    button.title = `${index + 1} · ${item.annotation_id}`;
    if (record.complete) {
      button.classList.add("complete");
    }
    if (record.revisit) {
      button.classList.add("revisit");
    }
    if (uncertain) {
      button.classList.add("uncertain");
    }
    if (index === currentIndex) {
      button.classList.add("current");
      button.setAttribute("aria-current", "true");
    }
    button.addEventListener("click", () => {
      navigateTo(index);
      $("#recordsDialog").close();
    });
    grid.append(button);
  });
  if (!grid.childElementCount) {
    const empty = document.createElement("p");
    empty.textContent = "Không có review ở bộ lọc này.";
    grid.append(empty);
  }
}

async function buildExportEnvelope(status, finalizedAt = null) {
  const payload = {
    workspace_schema_version: WORKSPACE_SCHEMA_VERSION,
    export_status: status,
    ui_version: UI_VERSION,
    workflow: deepClone(workflow),
    assignment_id: assignment.assignment_id,
    reference_id: assignment.reference_id,
    role: assignment.role,
    assignment_payload_sha256: assignment.assignment_payload_sha256,
    guideline_version: assignment.guideline.version,
    guideline_sha256: assignment.guideline.sha256,
    annotator_id: workspace.annotator_id.trim(),
    item_count: assignment.item_count,
    workspace_created_at: workspace.created_at,
    exported_at: nowUtc(),
    finalized_at: finalizedAt,
    records: assignment.records.map((item) =>
      deepClone(workspace.records[item.annotation_id]),
    ),
    audit_events: deepClone(workspace.audit_events),
  };
  return {
    schema_version: EXPORT_SCHEMA_VERSION,
    payload,
    payload_sha256: await sha256Text(canonicalJson(payload)),
  };
}

function downloadJson(value, filename) {
  const blob = new Blob(
    [`${JSON.stringify(value, null, 2)}\n`],
    { type: "application/json;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportDraft({ silent = false } = {}) {
  try {
    await flushSave();
    const envelope = await buildExportEnvelope("DRAFT", null);
    const stamp = new Date().toISOString().replaceAll(/[:.]/g, "-");
    downloadJson(
      envelope,
      `${assignment.assignment_id}-draft-${stamp}.json`,
    );
    if (!silent) {
      showToast("Đã tạo bản sao DRAFT. Hãy lưu file ở nơi an toàn.");
    }
    return true;
  } catch {
    return false;
  }
}

async function finalizeAssignment() {
  if (workspace.finalized_at) {
    const envelope = await buildExportEnvelope(
      "FINAL",
      workspace.finalized_at,
    );
    downloadJson(envelope, `${assignment.assignment_id}-FINAL.json`);
    return;
  }
  if (!workspace.annotator_id.trim()) {
    showToast("Hãy nhập mã người gán trước khi xuất FINAL.");
    $("#annotatorIdInput").focus();
    return;
  }
  const invalid = [];
  assignment.records.forEach((item, index) => {
    const record = workspace.records[item.annotation_id];
    const result = validateRecord(
      item.reviewContent,
      record,
      workspace.annotator_id,
    );
    if (!record.complete || !result.valid) {
      invalid.push(index + 1);
    }
  });
  if (invalid.length > 0) {
    showToast(
      `FINAL bị chặn: ${invalid.length} review chưa hoàn tất/hợp lệ. Review đầu tiên: ${invalid[0]}.`,
      7000,
    );
    navigateTo(invalid[0] - 1);
    return;
  }
  const accepted = await askConfirm({
    eyebrow: "Khóa assignment",
    title: `Xuất FINAL đủ ${assignment.item_count} review?`,
    message:
      "Sau khi khóa, UI chuyển sang chỉ đọc. File FINAL vẫn phải qua validator Python trước khi dùng làm human reference.",
    acceptText: "Khóa và xuất FINAL",
  });
  if (!accepted) {
    return;
  }
  const previousFinalizedAt = workspace.finalized_at;
  workspace.finalized_at = nowUtc();
  workspace.audit_events.push({
    event: "ASSIGNMENT_FINALIZED",
    at: workspace.finalized_at,
  });
  try {
    await flushSave();
    const envelope = await buildExportEnvelope(
      "FINAL",
      workspace.finalized_at,
    );
    downloadJson(envelope, `${assignment.assignment_id}-FINAL.json`);
    renderRecord();
    refreshHeader();
    $("#finalizeButton").textContent = "Tải lại FINAL";
    showToast("Assignment đã khóa. File FINAL đã được tạo.", 7000);
  } catch {
    workspace.finalized_at = previousFinalizedAt;
  }
}

async function verifyExportEnvelope(raw) {
  if (
    !exactKeys(raw, ["schema_version", "payload", "payload_sha256"]) ||
    raw.schema_version !== EXPORT_SCHEMA_VERSION ||
    !raw.payload ||
    typeof raw.payload !== "object"
  ) {
    throw new Error("File không đúng export schema.");
  }
  const actualHash = await sha256Text(canonicalJson(raw.payload));
  if (actualHash !== raw.payload_sha256) {
    throw new Error("Checksum export không khớp.");
  }
  const payload = raw.payload;
  if (!exactKeys(payload, EXPORT_PAYLOAD_KEYS)) {
    throw new Error("Export payload có field thiếu/thừa.");
  }
  rejectForbiddenKeys(payload);
  if (
    payload.assignment_id !== assignment.assignment_id ||
    payload.role !== assignment.role ||
    canonicalJson(payload.workflow) !== canonicalJson(workflow) ||
    payload.assignment_payload_sha256 !==
      assignment.assignment_payload_sha256 ||
    payload.guideline_sha256 !== assignment.guideline.sha256
  ) {
    throw new Error(
      "Bản sao thuộc assignment/role/workflow/guideline khác.",
    );
  }
  if (
    !Array.isArray(payload.records) ||
    payload.records.length !== assignment.item_count
  ) {
    throw new Error("Bản sao không đủ đúng số record.");
  }
  const byId = new Map(
    payload.records.map((record) => [record.annotation_id, record]),
  );
  if (byId.size !== payload.records.length) {
    throw new Error("Bản sao có annotation ID trùng.");
  }
  for (const item of assignment.records) {
    const record = byId.get(item.annotation_id);
    if (!record || record.review_text_sha256 !== item.review_text_sha256) {
      throw new Error("Bản sao có ID hoặc review hash không khớp.");
    }
    verifyWorkspaceRecordStructure(
      record,
      item,
      `Bản sao ${item.annotation_id}`,
    );
    for (const [revisionIndex, revision] of record.revisions.entries()) {
      const revisionRecord = {
        ...newRecordState(item.annotation_id, item.review_text_sha256),
        ...revision.annotation,
        read_complete: true,
      };
      const revisionResult = validateRecord(
        item.reviewContent,
        revisionRecord,
        payload.annotator_id,
      );
      if (!revisionResult.valid) {
        throw new Error(
          `Revision ${revisionIndex + 1} không hợp lệ ở ${item.annotation_id}.`,
        );
      }
    }
    if (record.complete) {
      const result = validateRecord(
        item.reviewContent,
        record,
        payload.annotator_id,
      );
      if (!result.valid) {
        throw new Error(
          `Bản sao có completed record không hợp lệ: ${item.annotation_id}.`,
        );
      }
    }
  }
  return payload;
}

async function importBackup(file) {
  if (!file) {
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showToast("File vượt giới hạn 8 MiB.");
    return;
  }
  try {
    const raw = JSON.parse(await file.text());
    const payload = await verifyExportEnvelope(raw);
    const accepted = await askConfirm({
      eyebrow: "Khôi phục có kiểm soát",
      title: "Thay bản nháp hiện tại?",
      message:
        "UI sẽ tự tải xuống một backup trước khi nhập. Không thể nhập chéo assignment, role hoặc workflow mode.",
      acceptText: "Sao lưu rồi khôi phục",
    });
    if (!accepted) {
      return;
    }
    const backedUp = await exportDraft({ silent: true });
    if (!backedUp) {
      showToast("Không tạo được backup hiện tại nên đã hủy khôi phục.");
      return;
    }
    workspace.annotator_id = String(payload.annotator_id || "");
    workspace.records = Object.fromEntries(
      payload.records.map((record) => [
        record.annotation_id,
        deepClone(record),
      ]),
    );
    workspace.audit_events = Array.isArray(payload.audit_events)
      ? deepClone(payload.audit_events)
      : [];
    workspace.audit_events.push({
      event: "BACKUP_IMPORTED",
      source_payload_sha256: raw.payload_sha256,
      at: nowUtc(),
    });
    workspace.finalized_at = payload.finalized_at || null;
    currentIndex = 0;
    workspace.current_index = 0;
    await flushSave();
    $("#annotatorIdInput").value = workspace.annotator_id;
    $("#finalizeButton").textContent = workspace.finalized_at
      ? "Tải lại FINAL"
      : "Xuất FINAL";
    renderRecord();
    refreshHeader();
    showToast("Đã khôi phục bản sao hợp lệ.", 6000);
  } catch (error) {
    showToast(`Từ chối bản sao: ${error.message}`, 9000);
  } finally {
    $("#importFileInput").value = "";
  }
}

function askConfirm({
  eyebrow = "Xác nhận",
  title,
  message,
  acceptText = "Xác nhận",
}) {
  const dialog = $("#confirmDialog");
  $("#confirmEyebrow").textContent = eyebrow;
  $("#confirmTitle").textContent = title;
  $("#confirmMessage").textContent = message;
  $("#confirmAcceptButton").textContent = acceptText;
  return new Promise((resolve) => {
    const finish = (value) => {
      $("#confirmAcceptButton").onclick = null;
      $("#confirmCancelButton").onclick = null;
      if (dialog.open) {
        dialog.close();
      }
      resolve(value);
    };
    $("#confirmAcceptButton").onclick = () => finish(true);
    $("#confirmCancelButton").onclick = () => finish(false);
    dialog.addEventListener(
      "cancel",
      (event) => {
        event.preventDefault();
        finish(false);
      },
      { once: true },
    );
    dialog.showModal();
  });
}

function dialogIsOpen() {
  return [...document.querySelectorAll("dialog")].some((dialog) => dialog.open);
}

function installStaticEventHandlers() {
  $("#retryButton").addEventListener("click", () => location.reload());
  $("#previousButton").addEventListener("click", () =>
    navigateTo(currentIndex - 1),
  );
  $("#nextButton").addEventListener("click", () =>
    navigateTo(currentIndex + 1),
  );
  $("#reviewText").addEventListener("mouseup", captureReviewSelection);
  $("#reviewText").addEventListener("keyup", captureReviewSelection);
  $("#clearSelectionButton").addEventListener("click", clearSelection);
  $("#readCompleteCheckbox").addEventListener("change", (event) => {
    currentRecordState().read_complete = event.target.checked;
    markRecordChanged("READ_CONFIRMATION_CHANGED");
  });
  $("#fillAbsentButton").addEventListener("click", async () => {
    const record = currentRecordState();
    if (!record.read_complete) {
      showToast("Chỉ dùng sau khi đã đọc toàn bộ review.");
      return;
    }
    const blanks = record.aspects.filter((row) => row.label === null);
    if (!blanks.length) {
      showToast("Không còn ô trống.");
      return;
    }
    const accepted = await askConfirm({
      eyebrow: "Thao tác có ghi audit",
      title: `Điền ${blanks.length} ô trống thành nhãn 2?`,
      message:
        "Chỉ xác nhận nếu bạn đã chủ động xét từng aspect. Không dùng để thay thế việc đọc review.",
      acceptText: "Tôi đã xét đủ, tiếp tục",
    });
    if (!accepted) {
      return;
    }
    for (const row of blanks) {
      row.label = 2;
      row.evidence = [];
    }
    workspace.audit_events.push({
      event: "BULK_FILL_ABSENT",
      annotation_id: record.annotation_id,
      affected_aspects: blanks.map((row) => row.aspect),
      at: nowUtc(),
    });
    markRecordChanged("BULK_FILL_ABSENT");
  });
  $("#notesInput").addEventListener("input", (event) => {
    currentRecordState().notes = event.target.value;
    markRecordChanged("NOTES_CHANGED", { rerender: false });
  });
  $("#revisitButton").addEventListener("click", () => {
    const record = currentRecordState();
    record.revisit = !record.revisit;
    record.updated_at = nowUtc();
    workspace.audit_events.push({
      event: record.revisit ? "REVISIT_SET" : "REVISIT_CLEARED",
      annotation_id: record.annotation_id,
      at: record.updated_at,
    });
    queueSave();
    renderRecordState();
  });
  $("#validateButton").addEventListener("click", validateCurrent);
  $("#completeButton").addEventListener("click", completeCurrent);
  $("#exportDraftButton").addEventListener("click", () => exportDraft());
  $("#finalizeButton").addEventListener("click", finalizeAssignment);
  $("#applySourceAButton").addEventListener("click", () =>
    applyAdjudicationSource("A"),
  );
  $("#applySourceBButton").addEventListener("click", () =>
    applyAdjudicationSource("B"),
  );
  $("#annotatorIdInput").addEventListener("input", (event) => {
    workspace.annotator_id = event.target.value.trim();
    queueSave();
  });
  $("#importButton").addEventListener("click", () =>
    $("#importFileInput").click(),
  );
  $("#importFileInput").addEventListener("change", (event) =>
    importBackup(event.target.files[0]),
  );

  for (const input of document.querySelectorAll(
    'input[name="annotationStatus"]',
  )) {
    input.addEventListener("change", async () => {
      if (!input.checked || workspace.finalized_at) {
        return;
      }
      const record = currentRecordState();
      const nextStatus = input.value;
      if (
        nextStatus === "REJECT_NON_REVIEW" &&
        record.annotation_status !== nextStatus
      ) {
        const accepted = await askConfirm({
          title: "Chuyển mẫu sang Không phải review?",
          message:
            "Toàn bộ chín nhãn, evidence và uncertainty cấp aspect sẽ được xóa. Bạn phải ghi lý do ngắn.",
          acceptText: "Chuyển và xóa nhãn",
        });
        if (!accepted) {
          renderRecord();
          return;
        }
        record.aspects.forEach((row) => {
          row.label = null;
          row.evidence = [];
          row.uncertainty_codes = [];
        });
        record.review_uncertainty_codes = ["NON_REVIEW"];
      } else if (record.annotation_status === "REJECT_NON_REVIEW") {
        record.review_uncertainty_codes =
          record.review_uncertainty_codes.filter(
            (code) => code !== "NON_REVIEW",
          );
      }
      record.annotation_status = nextStatus;
      markRecordChanged("ANNOTATION_STATUS_CHANGED");
    });
  }

  $("#openRecordsButton").addEventListener("click", () => {
    renderRecordGrid();
    $("#recordsDialog").showModal();
  });
  $("#openGuideButton").addEventListener("click", () =>
    $("#guideDialog").showModal(),
  );
  $("#openShortcutsButton").addEventListener("click", () =>
    $("#shortcutsDialog").showModal(),
  );
  document.querySelectorAll(".dialog-close").forEach((button) => {
    button.addEventListener("click", () => button.closest("dialog").close());
  });
  document.querySelectorAll(".filter-button").forEach((button) => {
    button.addEventListener("click", () => {
      recordFilter = button.dataset.filter;
      document
        .querySelectorAll(".filter-button")
        .forEach((node) => node.classList.toggle("active", node === button));
      renderRecordGrid();
    });
  });

  window.addEventListener("beforeunload", (event) => {
    if ($("#saveState").textContent !== "Đã lưu") {
      event.preventDefault();
    }
  });
  document.addEventListener("keydown", handleKeyboardShortcut);
}

function handleKeyboardShortcut(event) {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    completeCurrent();
    return;
  }
  if (event.altKey && /^[1-9]$/.test(event.key)) {
    event.preventDefault();
    const index = Number(event.key) - 1;
    focusedAspectIndex = index;
    const row = document.querySelector(
      `.aspect-row[data-aspect-index="${index}"]`,
    );
    row?.querySelector('input[type="radio"]:not(:disabled)')?.focus();
    row?.scrollIntoView({ block: "center", behavior: "smooth" });
    return;
  }
  const target = event.target;
  const editing =
    ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName) ||
    target?.isContentEditable;
  if (editing || dialogIsOpen() || event.ctrlKey || event.metaKey || event.altKey) {
    return;
  }
  if (event.key === "?") {
    event.preventDefault();
    $("#shortcutsDialog").showModal();
    return;
  }
  if (event.key.toLowerCase() === "g") {
    event.preventDefault();
    $("#revisitButton").click();
    return;
  }
  if (event.key.toLowerCase() === "e") {
    event.preventDefault();
    const row = currentRecordState().aspects[focusedAspectIndex];
    if (row.label === "1, -1") {
      showToast("Với Mixed, dùng nút + Positive hoặc + Negative.");
      return;
    }
    const polarity =
      row.label === 1
        ? "positive"
        : row.label === -1
          ? "negative"
          : row.label === 0
            ? "neutral"
            : null;
    if (polarity) {
      addEvidence(focusedAspectIndex, polarity);
    } else {
      showToast("Hãy chọn một nhãn mentioned trước.");
    }
    return;
  }
  const option = LABEL_OPTIONS.find(
    (candidate) => candidate.key.toLowerCase() === event.key.toLowerCase(),
  );
  if (option && !workspace.finalized_at) {
    event.preventDefault();
    setAspectLabel(focusedAspectIndex, option.value);
  }
}

async function initialize() {
  installStaticEventHandlers();
  try {
    if (!window.indexedDB || !window.crypto?.subtle) {
      throw new Error(
        "Browser thiếu IndexedDB hoặc Web Crypto. Hãy dùng Chrome/Edge hiện đại trên localhost.",
      );
    }
    const response = await fetch("/assignment.json", {
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!response.ok) {
      throw new Error(`Không tải được assignment: HTTP ${response.status}.`);
    }
    assignment = await verifyAssignment(await response.json());
    const workflowResponse = await fetch("/workflow.json", {
      cache: "no-store",
      credentials: "same-origin",
    });
    if (!workflowResponse.ok) {
      throw new Error(
        `Không tải được workflow: HTTP ${workflowResponse.status}.`,
      );
    }
    workflow = await verifyWorkflow(await workflowResponse.json());
    if (workflow.payload.suggestions_available) {
      const suggestionsResponse = await fetch("/suggestions.json", {
        cache: "no-store",
        credentials: "same-origin",
      });
      if (!suggestionsResponse.ok) {
        throw new Error(
          `Không tải được AI suggestions: HTTP ${suggestionsResponse.status}.`,
        );
      }
      suggestions = await verifySuggestions(
        await suggestionsResponse.json(),
      );
    }
    if (workflow.payload.adjudication_available) {
      const adjudicationResponse = await fetch("/adjudication.json", {
        cache: "no-store",
        credentials: "same-origin",
      });
      if (!adjudicationResponse.ok) {
        throw new Error(
          `Không tải được adjudication input: HTTP ${adjudicationResponse.status}.`,
        );
      }
      adjudication = await verifyAdjudication(
        await adjudicationResponse.json(),
      );
    }
    database = await openDatabase();
    const stored = await readStoredWorkspace();
    workspace = stored
      ? verifyStoredWorkspace(stored)
      : createWorkspace(suggestions);
    currentIndex = Math.min(
      Math.max(Number(workspace.current_index) || 0, 0),
      assignment.item_count - 1,
    );
    workspace.current_index = currentIndex;
    $("#annotatorIdInput").value = workspace.annotator_id;
    $("#finalizeButton").textContent = workspace.finalized_at
      ? "Tải lại FINAL"
      : "Xuất FINAL";
    if (workflow.payload.mode === "AI_ASSISTED_HUMAN_VERIFICATION") {
      $("#reviewModeBanner").hidden = false;
      $("#workflowBannerTitle").textContent =
        "Human-check có AI hỗ trợ";
      $("#workflowBannerText").textContent =
        "Nhãn và evidence ban đầu là gợi ý AI. Hãy đọc, xác nhận hoặc sửa từng review; output không phải double-blind gold.";
      $("#workflowEyebrow").textContent =
        "Human verification · AI-assisted";
      document.title = "Human-check AI Pre-annotations";
    } else if (workflow.payload.mode === "EXPERT_ADJUDICATION") {
      $("#reviewModeBanner").hidden = false;
      $("#reviewModeBanner").classList.add("adjudication");
      $("#workflowBannerTitle").textContent = "Expert adjudication";
      $("#workflowBannerText").textContent =
        "Nguồn A/B đã được khóa và IAA phải được công bố trước package này. Mọi disagreement cần quyết định expert.";
      $("#workflowEyebrow").textContent =
        "Expert resolution · A/B frozen";
      document.title = "ABSA Expert Adjudication";
    } else {
      $("#workflowEyebrow").textContent =
        "Human gold · double-blind";
      document.title = "Human ABSA Blind Annotation";
    }
    renderHeaderIdentity();
    renderRecord();
    refreshHeader();
    $("#bootScreen").hidden = true;
    $("#appShell").hidden = false;
    await persistState();
    if (navigator.storage?.persist) {
      const persisted = await navigator.storage.persist();
      if (!persisted) {
        showToast(
          "Browser chưa cấp persistent storage. Hãy sao lưu mỗi 25 review.",
          7000,
        );
      }
    }
  } catch (error) {
    showFatal(
      "Không thể mở phiên gán nhãn",
      error?.message || "Lỗi không xác định.",
    );
  }
}

initialize();
