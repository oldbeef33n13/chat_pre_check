const state = {
  config: null,
  source: "",
  selectedTemplateId: null,
  generated: null,
  generatedBatch: null,
  matchResult: null,
  search: "",
  batchMatchSource: createEmptyBatchSource(),
  batchMatchResult: null,
  batchGenerateSource: createEmptyBatchSource(),
  optimizationResult: null,
  inspector: {
    open: false,
    title: "",
    content: ""
  },
  llmSettings: {
    apiKey: "",
    baseUrl: "https://coding.dashscope.aliyuncs.com/v1",
    model: "qwen3-coder-plus",
    insecureSSL: false
  }
};

const els = {
  importFile: document.getElementById("import-file"),
  saveWorkspace: document.getElementById("save-workspace"),
  resetWorkspace: document.getElementById("reset-workspace"),
  exportConfig: document.getElementById("export-config"),
  templateSearch: document.getElementById("template-search"),
  templateList: document.getElementById("template-list"),
  workspaceMeta: document.getElementById("workspace-meta"),
  addTemplate: document.getElementById("add-template"),
  duplicateTemplate: document.getElementById("duplicate-template"),
  deleteTemplate: document.getElementById("delete-template"),
  showTemplateDetails: document.getElementById("show-template-details"),
  editorRoot: document.getElementById("editor-root"),
  generatorInput: document.getElementById("generator-input"),
  generatorResult: document.getElementById("generator-result"),
  generateTemplate: document.getElementById("generate-template"),
  batchGeneratorInput: document.getElementById("batch-generator-input"),
  batchGeneratorFile: document.getElementById("batch-generator-file"),
  parseBatchGenerator: document.getElementById("parse-batch-generator"),
  clearBatchGenerator: document.getElementById("clear-batch-generator"),
  batchGeneratorMeta: document.getElementById("batch-generator-meta"),
  generateTemplateBatch: document.getElementById("generate-template-batch"),
  batchGeneratorSummary: document.getElementById("batch-generator-summary"),
  batchGeneratorResult: document.getElementById("batch-generator-result"),
  llmApiKey: document.getElementById("llm-api-key"),
  llmBaseUrl: document.getElementById("llm-base-url"),
  llmModel: document.getElementById("llm-model"),
  llmInsecureSsl: document.getElementById("llm-insecure-ssl"),
  matchInput: document.getElementById("match-input"),
  scopeCurrentTemplate: document.getElementById("scope-current-template"),
  runMatch: document.getElementById("run-match"),
  matchResult: document.getElementById("match-result"),
  batchMatchInput: document.getElementById("batch-match-input"),
  batchMatchFile: document.getElementById("batch-match-file"),
  parseBatchMatch: document.getElementById("parse-batch-match"),
  clearBatchMatch: document.getElementById("clear-batch-match"),
  batchMatchMeta: document.getElementById("batch-match-meta"),
  scopeCurrentTemplateBatch: document.getElementById("scope-current-template-batch"),
  runBatchMatch: document.getElementById("run-batch-match"),
  runEvaluationCases: document.getElementById("run-evaluation-cases"),
  batchMatchSummary: document.getElementById("batch-match-summary"),
  batchMatchResult: document.getElementById("batch-match-result"),
  optimizeSource: document.getElementById("optimize-source"),
  optimizeTargetMetric: document.getElementById("optimize-target-metric"),
  optimizeStrategy: document.getElementById("optimize-strategy"),
  optimizeTargetValue: document.getElementById("optimize-target-value"),
  optimizeMaxRounds: document.getElementById("optimize-max-rounds"),
  optimizeMaxCandidates: document.getElementById("optimize-max-candidates"),
  runOptimization: document.getElementById("run-optimization"),
  applyOptimizedConfig: document.getElementById("apply-optimized-config"),
  optimizationSummary: document.getElementById("optimization-summary"),
  optimizationResult: document.getElementById("optimization-result"),
  inspectorModal: document.getElementById("inspector-modal"),
  inspectorTitle: document.getElementById("inspector-title"),
  inspectorContent: document.getElementById("inspector-content"),
  closeInspector: document.getElementById("close-inspector")
};

await bootstrap();

async function bootstrap() {
  hydrateLlmSettings();
  bindTopLevelEvents();
  const payload = await api("/api/workspace");
  state.config = payload.config;
  state.source = payload.source;
  state.llmSettings = {
    apiKey: localStorage.getItem("templateStudio.apiKey") || payload.llmDefaults.apiKey || "",
    baseUrl: localStorage.getItem("templateStudio.baseUrl") || payload.llmDefaults.baseUrl,
    model: localStorage.getItem("templateStudio.model") || payload.llmDefaults.model,
    insecureSSL: readBooleanSetting("templateStudio.insecureSSL", payload.llmDefaults.insecureSSL)
  };
  if (state.config.templates.length) {
    state.selectedTemplateId = state.config.templates[0].template_id;
  }
  renderAll();
}

function bindTopLevelEvents() {
  els.importFile.addEventListener("change", handleImportFile);
  els.saveWorkspace.addEventListener("click", handleSaveWorkspace);
  els.resetWorkspace.addEventListener("click", handleResetWorkspace);
  els.exportConfig.addEventListener("click", handleExportConfig);
  els.templateSearch.addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderTemplateList();
  });
  els.addTemplate.addEventListener("click", () => {
    const template = createBlankTemplate();
    state.config.templates.unshift(template);
    state.selectedTemplateId = template.template_id;
    renderAll();
  });
  els.duplicateTemplate.addEventListener("click", () => {
    const template = getCurrentTemplate();
    if (!template) {
      return;
    }
    const next = structuredClone(template);
    next.template_id = `${template.template_id}.copy`;
    state.config.templates.unshift(next);
    state.selectedTemplateId = next.template_id;
    renderAll();
  });
  els.deleteTemplate.addEventListener("click", () => {
    const template = getCurrentTemplate();
    if (!template) {
      return;
    }
    if (!window.confirm(`确定删除模板 ${template.template_id} 吗？`)) {
      return;
    }
    state.config.templates = state.config.templates.filter((item) => item.template_id !== template.template_id);
    state.selectedTemplateId = state.config.templates[0]?.template_id || null;
    closeInspector();
    renderAll();
  });
  els.showTemplateDetails.addEventListener("click", () => {
    const template = getCurrentTemplate();
    if (!template) {
      return;
    }
    openInspector(`模板详情：${template.template_id}`, formatJson(template));
  });
  els.generateTemplate.addEventListener("click", handleGenerateTemplate);
  els.parseBatchGenerator.addEventListener("click", () =>
    loadBatchSourceFromText({
      kind: "generate",
      content: els.batchGeneratorInput.value,
      filename: "inline-batch-generate.txt"
    })
  );
  els.batchGeneratorFile.addEventListener("change", (event) => handleBatchFile(event, "generate"));
  els.clearBatchGenerator.addEventListener("click", clearBatchGeneratorState);
  els.generateTemplateBatch.addEventListener("click", handleGenerateTemplateBatch);
  els.llmApiKey.addEventListener("change", persistLlmSettings);
  els.llmBaseUrl.addEventListener("change", persistLlmSettings);
  els.llmModel.addEventListener("change", persistLlmSettings);
  els.llmInsecureSsl.addEventListener("change", persistLlmSettings);
  els.runMatch.addEventListener("click", handleRunMatch);
  els.parseBatchMatch.addEventListener("click", () =>
    loadBatchSourceFromText({
      kind: "match",
      content: els.batchMatchInput.value,
      filename: "inline-batch-match.txt"
    })
  );
  els.batchMatchFile.addEventListener("change", (event) => handleBatchFile(event, "match"));
  els.clearBatchMatch.addEventListener("click", clearBatchMatchState);
  els.runBatchMatch.addEventListener("click", () => handleRunBatchMatch());
  els.runEvaluationCases.addEventListener("click", handleRunEvaluationCases);
  els.runOptimization.addEventListener("click", handleRunOptimization);
  els.applyOptimizedConfig.addEventListener("click", handleApplyOptimizedConfig);
  els.closeInspector.addEventListener("click", closeInspector);
  els.inspectorModal.addEventListener("click", (event) => {
    if (event.target === els.inspectorModal) {
      closeInspector();
    }
  });
}

function renderAll() {
  renderWorkspaceMeta();
  renderTemplateList();
  renderGeneratorResult();
  renderBatchGenerateSource();
  renderBatchGenerateOutput();
  renderEditor();
  renderMatchResult();
  renderBatchMatchSource();
  renderBatchMatchOutput();
  renderOptimizationOutput();
  renderInspector();
  els.llmApiKey.value = state.llmSettings.apiKey;
  els.llmBaseUrl.value = state.llmSettings.baseUrl;
  els.llmModel.value = state.llmSettings.model;
  els.llmInsecureSsl.checked = Boolean(state.llmSettings.insecureSSL);
  const hasTemplate = Boolean(getCurrentTemplate());
  els.duplicateTemplate.disabled = !hasTemplate;
  els.deleteTemplate.disabled = !hasTemplate;
  els.showTemplateDetails.disabled = !hasTemplate;
  els.applyOptimizedConfig.disabled = !state.optimizationResult?.optimized_config;
}

function renderWorkspaceMeta() {
  const count = state.config?.templates?.length || 0;
  els.workspaceMeta.textContent = `来源：${state.source} · 模板数：${count}`;
}

function renderTemplateList() {
  const templates = (state.config?.templates || []).filter((template) => {
    if (!state.search) {
      return true;
    }
    return [template.template_id, template.description].some((value) =>
      String(value || "").toLowerCase().includes(state.search)
    );
  });
  els.templateList.innerHTML = templates.length
    ? templates
        .map(
          (template) => `
            <article class="template-item ${template.template_id === state.selectedTemplateId ? "active" : ""}" data-template-id="${escapeHtml(template.template_id)}">
              <h3>${escapeHtml(template.template_id)}</h3>
              <p>${escapeHtml(template.description || "暂无描述")}</p>
            </article>
          `
        )
        .join("")
    : `<div class="muted">没有匹配到模板。</div>`;

  for (const item of els.templateList.querySelectorAll(".template-item")) {
    item.addEventListener("click", () => {
      state.selectedTemplateId = item.dataset.templateId;
      renderAll();
    });
  }
}

function renderEditor() {
  const template = getCurrentTemplate();
  if (!template) {
    els.editorRoot.innerHTML = `<div class="muted">当前没有模板，请先导入模板文件或新建模板。</div>`;
    return;
  }

  els.editorRoot.innerHTML = `
    <div class="editor-form">
      <div class="field-grid">
        <label class="field">
          <span>template_id</span>
          <input id="field-template-id" type="text" value="${escapeHtml(template.template_id)}" />
        </label>
        <label class="field">
          <span>query_mode</span>
          <input id="field-query-mode" type="text" value="${escapeHtml(template.query_mode || "metric_query")}" />
        </label>
        <label class="field full-span">
          <span>description</span>
          <textarea id="field-description" rows="3">${escapeHtml(template.description || "")}</textarea>
        </label>
        <label class="field full-span">
          <span>utterances（每行一条）</span>
          <textarea id="field-utterances" rows="5">${escapeHtml((template.utterances || []).join("\n"))}</textarea>
        </label>
        <label class="field">
          <span>required_slots（逗号分隔）</span>
          <input id="field-required-slots" type="text" value="${escapeHtml((template.required_slots || []).join(", "))}" />
        </label>
        <label class="field">
          <span>optional_slots（逗号分隔）</span>
          <input id="field-optional-slots" type="text" value="${escapeHtml((template.optional_slots || []).join(", "))}" />
        </label>
        <label class="field full-span">
          <span>must_terms（每行一组，同组内用 | 分隔）</span>
          <textarea id="field-must-terms" rows="4">${escapeHtml(formatMustTerms(template.must_terms))}</textarea>
        </label>
        <label class="field full-span">
          <span>negative_terms（逗号分隔）</span>
          <input id="field-negative-terms" type="text" value="${escapeHtml((template.negative_terms || []).join(", "))}" />
        </label>
      </div>

      <div class="hint-strip">
        <span class="badge">紧凑编辑模式</span>
        <p class="muted">首页先做测试，模板原始 JSON 需要时再点右上角查看；高级字段放在下面折叠区，避免编辑区过长。</p>
      </div>

      <details class="details-block">
        <summary>高级 JSON：约束与抽取</summary>
        <div class="details-body field-grid">
          <label class="field full-span">
            <span>slot_constraints（JSON）</span>
            <textarea id="field-slot-constraints" class="json-editor" rows="8">${escapeHtml(formatJson(template.slot_constraints))}</textarea>
          </label>
          <label class="field full-span">
            <span>slot_extractors（JSON）</span>
            <textarea id="field-slot-extractors" class="json-editor" rows="12">${escapeHtml(formatJson(template.slot_extractors))}</textarea>
          </label>
          <label class="field full-span">
            <span>slot_defaults（JSON）</span>
            <textarea id="field-slot-defaults" class="json-editor" rows="6">${escapeHtml(formatJson(template.slot_defaults || {}))}</textarea>
          </label>
          <label class="field full-span">
            <span>derived_slots（JSON）</span>
            <textarea id="field-derived-slots" class="json-editor" rows="8">${escapeHtml(formatJson(template.derived_slots || []))}</textarea>
          </label>
          <label class="field full-span">
            <span>slot_validations（JSON）</span>
            <textarea id="field-slot-validations" class="json-editor" rows="6">${escapeHtml(formatJson(template.slot_validations || {}))}</textarea>
          </label>
          <label class="field full-span">
            <span>llm_slot_extraction（JSON）</span>
            <textarea id="field-llm-slot-extraction" class="json-editor" rows="7">${escapeHtml(formatJson(template.llm_slot_extraction))}</textarea>
          </label>
        </div>
      </details>

      <details class="details-block">
        <summary>高级 JSON：metadata</summary>
        <div class="details-body">
          <label class="field full-span">
            <span>metadata（JSON）</span>
            <textarea id="field-metadata" class="json-editor" rows="6">${escapeHtml(formatJson(template.metadata))}</textarea>
          </label>
        </div>
      </details>
    </div>
  `;

  bindEditorField("field-template-id", (value) => {
    const oldId = template.template_id;
    template.template_id = value || `template.generated.${Date.now()}`;
    if (state.selectedTemplateId === oldId) {
      state.selectedTemplateId = template.template_id;
    }
    renderTemplateList();
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-query-mode", (value) => {
    template.query_mode = value || "metric_query";
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-description", (value) => {
    template.description = value;
    renderTemplateList();
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-utterances", (value) => {
    template.utterances = splitLines(value);
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-required-slots", (value) => {
    template.required_slots = splitCommaList(value);
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-optional-slots", (value) => {
    template.optional_slots = splitCommaList(value);
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-must-terms", (value) => {
    template.must_terms = parseMustTerms(value);
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindEditorField("field-negative-terms", (value) => {
    template.negative_terms = splitCommaList(value);
    refreshInspectorIfShowingCurrentTemplate();
  });
  bindJsonField("field-slot-constraints", (json) => {
    template.slot_constraints = json;
  });
  bindJsonField("field-slot-extractors", (json) => {
    template.slot_extractors = json;
  });
  bindJsonField("field-slot-defaults", (json) => {
    template.slot_defaults = json;
  });
  bindJsonField("field-derived-slots", (json) => {
    template.derived_slots = json;
  });
  bindJsonField("field-slot-validations", (json) => {
    template.slot_validations = json;
  });
  bindJsonField("field-llm-slot-extraction", (json) => {
    template.llm_slot_extraction = json;
  });
  bindJsonField("field-metadata", (json) => {
    template.metadata = json;
  });
}

function renderGeneratorResult() {
  if (!state.generated) {
    els.generatorResult.className = "generator-result empty";
    els.generatorResult.textContent = "尚未生成模板建议。";
    return;
  }
  const notes = Array.isArray(state.generated.analysis.notes) ? state.generated.analysis.notes : [];
  const rationale = Array.isArray(state.generated.analysis.design_rationale)
    ? state.generated.analysis.design_rationale
    : [];
  els.generatorResult.className = "generator-result";
  els.generatorResult.innerHTML = `
    <div class="pill-row">
      <span class="pill">entity: ${escapeHtml(state.generated.analysis.entity || "-")}</span>
      <span class="pill">metric: ${escapeHtml(state.generated.analysis.metric || "-")}</span>
      <span class="pill">query_operator: ${escapeHtml(state.generated.analysis.query_operator || "-")}</span>
    </div>
    <p class="muted">${escapeHtml(state.generated.analysis.intent || "模型已返回模板建议。")}</p>
    ${state.generated.analysis.template_strategy ? `<p class="muted">设计策略：${escapeHtml(state.generated.analysis.template_strategy)}</p>` : ""}
    ${notes.length ? `<ul class="hint-list">${notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("")}</ul>` : ""}
    ${rationale.length ? `<h3>为什么这样设计</h3><ul class="hint-list">${rationale.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
    <div class="inline-actions">
      <button id="apply-generated-current">覆盖当前模板</button>
      <button id="apply-generated-new" class="secondary">新增为新模板</button>
      <button id="view-generated-json" class="secondary">查看 JSON</button>
    </div>
  `;

  document.getElementById("apply-generated-current")?.addEventListener("click", () => {
    const current = getCurrentTemplate();
    if (!current) {
      return;
    }
    const oldId = current.template_id;
    Object.assign(current, structuredClone(state.generated.template));
    if (state.selectedTemplateId === oldId) {
      state.selectedTemplateId = current.template_id;
    }
    renderAll();
  });
  document.getElementById("apply-generated-new")?.addEventListener("click", () => {
    const inserted = insertTemplateWithUniqueId(state.generated.template, { select: true });
    state.selectedTemplateId = inserted.template_id;
    renderAll();
  });
  document.getElementById("view-generated-json")?.addEventListener("click", () => {
    openInspector("生成结果 JSON", formatJson(state.generated.template));
  });
}

function renderBatchGenerateSource() {
  renderBatchSourceMeta({
    element: els.batchGeneratorMeta,
    source: state.batchGenerateSource,
    emptyMessage: "尚未加载批量模板生成输入，支持 txt / json / jsonl，字段可用 text / query / question。"
  });
}

function renderBatchGenerateOutput() {
  if (!state.generatedBatch) {
    els.batchGeneratorSummary.className = "summary-panel empty";
    els.batchGeneratorSummary.textContent = "尚未执行批量模板生成。";
    els.batchGeneratorResult.className = "result-panel empty";
    els.batchGeneratorResult.textContent = "批量生成结果会显示在这里。";
    return;
  }

  els.batchGeneratorSummary.className = "summary-panel";
  els.batchGeneratorSummary.innerHTML = `
    <div class="pill-row">
      <span class="pill">total: ${escapeHtml(state.generatedBatch.total)}</span>
      <span class="pill">success: ${escapeHtml(state.generatedBatch.success_count)}</span>
      <span class="pill">failure: ${escapeHtml(state.generatedBatch.failure_count)}</span>
    </div>
    <div class="inline-actions">
      <button id="apply-generated-batch" class="secondary" ${state.generatedBatch.success_count ? "" : "disabled"}>批量加入模板区</button>
    </div>
  `;

  els.batchGeneratorResult.className = "result-panel";
  els.batchGeneratorResult.innerHTML = `
    <div class="batch-card-list">
      ${state.generatedBatch.results
        .map((entry, index) =>
          entry.ok
            ? `
              <article class="batch-card">
                <div class="batch-card-head">
                  <div>
                    <strong>#${index + 1}</strong>
                    <span class="pill success">已生成</span>
                  </div>
                  <div class="inline-actions">
                    <button data-add-generated-index="${index}" class="secondary">新增模板</button>
                    <button data-view-generated-index="${index}" class="secondary">查看 JSON</button>
                  </div>
                </div>
                <p>${escapeHtml(entry.item.text)}</p>
                <p class="muted">${escapeHtml(entry.output.analysis.intent || entry.output.template.description || "模型已返回模板建议。")}</p>
                ${renderRationaleList(entry.output.analysis.design_rationale)}
              </article>
            `
            : `
              <article class="batch-card batch-card-error">
                <div class="batch-card-head">
                  <div>
                    <strong>#${index + 1}</strong>
                    <span class="pill danger">失败</span>
                  </div>
                </div>
                <p>${escapeHtml(entry.item.text)}</p>
                <p class="muted">${escapeHtml(entry.error || "未知错误")}</p>
              </article>
            `
        )
        .join("")}
    </div>
  `;

  document.getElementById("apply-generated-batch")?.addEventListener("click", () => {
    const successful = state.generatedBatch.results.filter((entry) => entry.ok).map((entry) => entry.output.template);
    if (!successful.length) {
      return;
    }
    let firstInserted = null;
    for (const template of successful) {
      const inserted = insertTemplateWithUniqueId(template, { select: false });
      if (!firstInserted) {
        firstInserted = inserted;
      }
    }
    if (firstInserted) {
      state.selectedTemplateId = firstInserted.template_id;
    }
    renderAll();
  });

  for (const button of els.batchGeneratorResult.querySelectorAll("[data-add-generated-index]")) {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.addGeneratedIndex);
      const entry = state.generatedBatch.results[index];
      if (!entry?.ok) {
        return;
      }
      const inserted = insertTemplateWithUniqueId(entry.output.template, { select: true });
      state.selectedTemplateId = inserted.template_id;
      renderAll();
    });
  }

  for (const button of els.batchGeneratorResult.querySelectorAll("[data-view-generated-index]")) {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.viewGeneratedIndex);
      const entry = state.generatedBatch.results[index];
      if (!entry?.ok) {
        return;
      }
      openInspector(`批量生成结果 #${index + 1}`, formatJson(entry.output.template));
    });
  }
}

function renderMatchResult() {
  if (!state.matchResult) {
    els.matchResult.className = "result-panel empty";
    els.matchResult.textContent = "尚未执行单条匹配测试。";
    return;
  }
  const result = state.matchResult;
  const topCandidates = result.trace?.top_candidates || [];
  const rewriteTrace = result.trace?.rewrite_trace || null;
  els.matchResult.className = "result-panel";
  els.matchResult.innerHTML = `
    <div class="pill-row">
      <span class="pill">template_id: ${escapeHtml(String(result.template_id))}</span>
      <span class="pill">status: ${escapeHtml(result.status)}</span>
      <span class="pill">score: ${escapeHtml(Number(result.score || 0).toFixed(4))}</span>
      <span class="pill">query_mode: ${escapeHtml(result.query_mode || "-")}</span>
    </div>
    <div class="inline-actions">
      <button id="view-single-match-details" class="secondary">查看完整结果</button>
    </div>
    <h3>前置改写</h3>
    <pre class="raw-preview">${escapeHtml(formatJson(rewriteTrace || {}))}</pre>
    <h3>抽取槽位</h3>
    <pre class="raw-preview">${escapeHtml(formatJson(result.slots || {}))}</pre>
    <h3>Top Candidates</h3>
    <pre class="raw-preview">${escapeHtml(formatJson(topCandidates))}</pre>
  `;

  document.getElementById("view-single-match-details")?.addEventListener("click", () => {
    openInspector("单条匹配完整结果", formatJson(result));
  });
}

function renderBatchMatchSource() {
  renderBatchSourceMeta({
    element: els.batchMatchMeta,
    source: state.batchMatchSource,
    emptyMessage: "尚未加载批量测试输入，支持 txt / json / jsonl，也支持直接加载仓库评测集。"
  });
}

function renderBatchMatchOutput() {
  if (!state.batchMatchResult) {
    els.batchMatchSummary.className = "summary-panel empty";
    els.batchMatchSummary.textContent = "尚未执行批量测试。";
    els.batchMatchResult.className = "result-panel empty";
    els.batchMatchResult.textContent = "批量测试结果会显示在这里。";
    return;
  }

  const summary = state.batchMatchResult.summary;
  els.batchMatchSummary.className = "summary-panel";
  els.batchMatchSummary.innerHTML = `
    <div class="pill-row">
      <span class="pill">total: ${escapeHtml(summary.total)}</span>
      <span class="pill">matched: ${escapeHtml(summary.status_counts.matched || 0)}</span>
      <span class="pill">partial: ${escapeHtml(summary.status_counts.partial || 0)}</span>
      <span class="pill">unmatched: ${escapeHtml(summary.status_counts.unmatched || 0)}</span>
      <span class="pill">evaluated: ${escapeHtml(summary.evaluated_count || 0)}</span>
      <span class="pill">pass: ${escapeHtml(summary.pass_count || 0)}</span>
      <span class="pill">fail: ${escapeHtml(summary.fail_count || 0)}</span>
      ${
        summary.pass_rate == null
          ? ""
          : `<span class="pill">pass_rate: ${escapeHtml((summary.pass_rate * 100).toFixed(1))}%</span>`
      }
    </div>
  `;

  els.batchMatchResult.className = "result-panel table-panel";
  els.batchMatchResult.innerHTML = `
    <div class="table-wrap">
      <table class="data-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Query</th>
            <th>期望</th>
            <th>实际</th>
            <th>得分</th>
            <th>评估</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${state.batchMatchResult.results
            .map((entry, index) => {
              const evaluationClass = entry.evaluation.evaluated
                ? entry.evaluation.pass
                  ? "row-pass"
                  : "row-fail"
                : "";
              const expected = formatExpectedLabel(entry.item);
              const actual = `${entry.result.template_id} / ${entry.result.status}`;
              const evaluation = entry.evaluation.evaluated
                ? entry.evaluation.pass
                  ? "通过"
                  : `失败：${entry.evaluation.message}`
                : "观测";
              return `
                <tr class="${evaluationClass}">
                  <td>${index + 1}</td>
                  <td class="query-cell">${escapeHtml(entry.item.text)}</td>
                  <td>${escapeHtml(expected)}</td>
                  <td>${escapeHtml(actual)}</td>
                  <td>${escapeHtml(Number(entry.result.score || 0).toFixed(4))}</td>
                  <td>${escapeHtml(evaluation)}</td>
                  <td><button data-view-batch-result="${index}" class="secondary compact">详情</button></td>
                </tr>
              `;
            })
            .join("")}
        </tbody>
      </table>
    </div>
  `;

  for (const button of els.batchMatchResult.querySelectorAll("[data-view-batch-result]")) {
    button.addEventListener("click", () => {
      const index = Number(button.dataset.viewBatchResult);
      const entry = state.batchMatchResult.results[index];
      openInspector(`批量测试结果 #${index + 1}`, formatJson(entry));
    });
  }
}

function renderOptimizationOutput() {
  if (!state.optimizationResult) {
    els.optimizationSummary.className = "summary-panel empty";
    els.optimizationSummary.textContent = "尚未执行持续优化。";
    els.optimizationResult.className = "result-panel empty";
    els.optimizationResult.textContent = "持续优化的轮次记录会显示在这里。";
    return;
  }

  const result = state.optimizationResult;
  const initialMetric = result.initial?.primary_value;
  const finalMetric = result.final?.primary_value;
  els.optimizationSummary.className = "summary-panel";
  els.optimizationSummary.innerHTML = `
    <div class="pill-row">
      <span class="pill">metric: ${escapeHtml(result.goal.metric)}</span>
      <span class="pill">strategy: ${escapeHtml(result.goal.strategy || "balanced")}</span>
      <span class="pill">target: ${escapeHtml(Number(result.goal.target_value).toFixed(2))}</span>
      <span class="pill">initial: ${escapeHtml(formatMetric(initialMetric))}</span>
      <span class="pill">final: ${escapeHtml(formatMetric(finalMetric))}</span>
      <span class="pill">${result.reached ? "已达标" : "未达标"}</span>
    </div>
    <div class="inline-actions">
      <button id="view-optimized-config" class="secondary">查看最终配置 JSON</button>
    </div>
  `;

  els.optimizationResult.className = "result-panel";
  els.optimizationResult.innerHTML = `
    <div class="batch-card-list">
      ${result.history.length
        ? result.history
            .map(
              (round) => `
                <article class="batch-card">
                  <div class="batch-card-head">
                    <div>
                      <strong>Round ${round.round}</strong>
                      <span class="pill">${escapeHtml(round.accepted_count)} accepted / ${escapeHtml(round.attempted_count)} attempted</span>
                    </div>
                    <span class="pill">${escapeHtml(round.stop_reason || "继续推进")}</span>
                  </div>
                  <p class="muted">before ${escapeHtml(formatMetric(round.before.primary_value))} -> after ${escapeHtml(formatMetric(round.after.primary_value))}</p>
                  ${
                    round.actions.length
                      ? `<div class="action-list">${round.actions
                          .map(
                            (action, index) => `
                              <article class="action-card ${action.accepted ? "action-accepted" : "action-rejected"}">
                                <div class="batch-card-head">
                                  <div>
                                    <strong>#${index + 1} ${escapeHtml(action.proposal_template_id || "(no id)")}</strong>
                                    <span class="pill ${action.accepted ? "success" : "danger"}">${action.accepted ? "accepted" : "rejected"}</span>
                                  </div>
                                  <button data-view-optimization-action="${round.round}-${index}" class="secondary compact">详情</button>
                                </div>
                                <p>${escapeHtml(action.text)}</p>
                                <p class="muted">${escapeHtml(action.strategy || action.mode)}</p>
                              </article>
                            `
                          )
                          .join("")}</div>`
                      : `<p class="muted">这一轮没有找到可推进的候选。</p>`
                  }
                </article>
              `
            )
            .join("")
        : `<div class="muted">没有轮次记录。</div>`}
    </div>
  `;

  document.getElementById("view-optimized-config")?.addEventListener("click", () => {
    openInspector("持续优化后的配置", formatJson(result.optimized_config));
  });

  for (const button of els.optimizationResult.querySelectorAll("[data-view-optimization-action]")) {
    button.addEventListener("click", () => {
      const [roundIndex, actionIndex] = String(button.dataset.viewOptimizationAction || "").split("-");
      const round = result.history.find((item) => String(item.round) === roundIndex);
      const action = round?.actions?.[Number(actionIndex)];
      if (!action) {
        return;
      }
      openInspector(`优化动作 Round ${roundIndex}`, formatJson(action));
    });
  }
}

function renderInspector() {
  els.inspectorModal.classList.toggle("hidden", !state.inspector.open);
  els.inspectorTitle.textContent = state.inspector.title || "详情";
  els.inspectorContent.textContent = state.inspector.content || "";
}

function bindEditorField(id, onCommit) {
  const element = document.getElementById(id);
  if (!element) {
    return;
  }
  element.addEventListener("change", (event) => onCommit(event.target.value));
}

function bindJsonField(id, onCommit) {
  const element = document.getElementById(id);
  if (!element) {
    return;
  }
  element.addEventListener("change", (event) => {
    try {
      const parsed = JSON.parse(event.target.value || "{}");
      onCommit(parsed);
      refreshInspectorIfShowingCurrentTemplate();
    } catch (error) {
      window.alert(`JSON 解析失败：${error.message}`);
      event.target.focus();
    }
  });
}

async function handleImportFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const text = await file.text();
  try {
    const config = JSON.parse(text);
    const payload = await api("/api/workspace/save", {
      method: "POST",
      body: JSON.stringify({ config })
    });
    state.config = payload.config;
    state.source = "workspace";
    state.selectedTemplateId = state.config.templates[0]?.template_id || null;
    state.generated = null;
    state.generatedBatch = null;
    state.matchResult = null;
    state.batchMatchResult = null;
    state.optimizationResult = null;
    renderAll();
  } catch (error) {
    window.alert(`导入失败：${error.message}`);
  } finally {
    event.target.value = "";
  }
}

async function handleSaveWorkspace() {
  const payload = await api("/api/workspace/save", {
    method: "POST",
    body: JSON.stringify({ config: state.config })
  });
  state.config = payload.config;
  state.source = "workspace";
  renderWorkspaceMeta();
  window.alert(`工作区已保存到 ${payload.workspacePath}`);
}

async function handleResetWorkspace() {
  if (!window.confirm("确定要恢复默认模板吗？当前未保存修改会丢失。")) {
    return;
  }
  const payload = await api("/api/workspace/reset", { method: "POST" });
  state.config = payload.config;
  state.source = payload.source;
  state.selectedTemplateId = state.config.templates[0]?.template_id || null;
  state.generated = null;
  state.generatedBatch = null;
  state.matchResult = null;
  state.batchMatchResult = null;
  state.optimizationResult = null;
  closeInspector();
  renderAll();
}

function handleExportConfig() {
  const blob = new Blob([`${formatJson(state.config)}\n`], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "templates.studio.export.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

async function handleGenerateTemplate() {
  const text = els.generatorInput.value.trim();
  if (!text) {
    window.alert("请先输入一句要抽象成模板的话。");
    return;
  }
  syncLlmSettingsFromInputs();
  els.generateTemplate.disabled = true;
  els.generateTemplate.textContent = "生成中...";
  try {
    const payload = await api("/api/llm/generate-template", {
      method: "POST",
      body: JSON.stringify({
        text,
        config: state.config,
        settings: state.llmSettings
      })
    });
    state.generated = payload;
    renderGeneratorResult();
  } catch (error) {
    window.alert(`模板生成失败：${error.message}`);
  } finally {
    els.generateTemplate.disabled = false;
    els.generateTemplate.textContent = "生成模板建议";
  }
}

async function handleGenerateTemplateBatch() {
  if (!state.batchGenerateSource.items.length) {
    window.alert("请先加载批量模板生成输入。");
    return;
  }
  syncLlmSettingsFromInputs();
  els.generateTemplateBatch.disabled = true;
  els.generateTemplateBatch.textContent = "批量生成中...";
  try {
    const payload = await api("/api/llm/generate-template/batch", {
      method: "POST",
      body: JSON.stringify({
        items: state.batchGenerateSource.items,
        config: state.config,
        settings: state.llmSettings
      })
    });
    state.generatedBatch = payload;
    renderBatchGenerateOutput();
  } catch (error) {
    window.alert(`批量模板生成失败：${error.message}`);
  } finally {
    els.generateTemplateBatch.disabled = false;
    els.generateTemplateBatch.textContent = "批量生成模板建议";
  }
}

async function handleRunMatch() {
  const text = els.matchInput.value.trim();
  if (!text) {
    window.alert("请先输入要测试的 query。");
    return;
  }
  els.runMatch.disabled = true;
  els.runMatch.textContent = "测试中...";
  try {
    const templateIds = els.scopeCurrentTemplate.checked && state.selectedTemplateId ? [state.selectedTemplateId] : null;
    const payload = await api("/api/match", {
      method: "POST",
      body: JSON.stringify({
        text,
        config: state.config,
        templateIds
      })
    });
    state.matchResult = payload.result;
    renderMatchResult();
  } catch (error) {
    window.alert(`匹配测试失败：${error.message}`);
  } finally {
    els.runMatch.disabled = false;
    els.runMatch.textContent = "运行单条测试";
  }
}

async function handleRunBatchMatch(items = state.batchMatchSource.items) {
  if (!items.length) {
    window.alert("请先加载批量测试输入。");
    return;
  }
  els.runBatchMatch.disabled = true;
  els.runBatchMatch.textContent = "批量测试中...";
  try {
    const templateIds =
      els.scopeCurrentTemplateBatch.checked && state.selectedTemplateId ? [state.selectedTemplateId] : null;
    const payload = await api("/api/match/batch", {
      method: "POST",
      body: JSON.stringify({
        items,
        config: state.config,
        templateIds
      })
    });
    state.batchMatchResult = payload;
    renderBatchMatchOutput();
  } catch (error) {
    window.alert(`批量测试失败：${error.message}`);
  } finally {
    els.runBatchMatch.disabled = false;
    els.runBatchMatch.textContent = "一键批量测试";
  }
}

async function handleRunEvaluationCases() {
  els.runEvaluationCases.disabled = true;
  els.runEvaluationCases.textContent = "加载并运行中...";
  try {
    const payload = await api("/api/examples/evaluation-cases");
    state.batchMatchSource = {
      items: payload.items || [],
      format: payload.format || "",
      source_name: payload.source_name || "",
      warnings: payload.warnings || []
    };
    renderBatchMatchSource();
    await handleRunBatchMatch(state.batchMatchSource.items);
  } catch (error) {
    window.alert(`加载仓库评测集失败：${error.message}`);
  } finally {
    els.runEvaluationCases.disabled = false;
    els.runEvaluationCases.textContent = "一键跑仓库评测集";
  }
}

async function handleRunOptimization() {
  const items = getOptimizationItems();
  if (!items.length) {
    window.alert("请先在批量测试区或批量生成区加载数据，再执行持续优化。");
    return;
  }
  syncLlmSettingsFromInputs();
  els.runOptimization.disabled = true;
  els.runOptimization.textContent = "优化中...";
  try {
    const payload = await api("/api/optimize/run", {
      method: "POST",
      body: JSON.stringify({
        items,
        config: state.config,
        settings: state.llmSettings,
        targetMetric: els.optimizeTargetMetric.value,
        optimizationStrategy: els.optimizeStrategy.value,
        targetValue: Number(els.optimizeTargetValue.value || 0.9),
        maxRounds: Number(els.optimizeMaxRounds.value || 3),
        maxCandidatesPerRound: Number(els.optimizeMaxCandidates.value || 5)
      })
    });
    state.optimizationResult = payload;
    renderOptimizationOutput();
  } catch (error) {
    window.alert(`持续优化失败：${error.message}`);
  } finally {
    els.runOptimization.disabled = false;
    els.runOptimization.textContent = "开始持续优化";
  }
}

function handleApplyOptimizedConfig() {
  const optimized = state.optimizationResult?.optimized_config;
  if (!optimized) {
    window.alert("当前没有可应用的优化结果。");
    return;
  }
  state.config = optimized;
  state.source = "optimized-session";
  state.selectedTemplateId = state.config.templates[0]?.template_id || state.selectedTemplateId;
  renderAll();
}

async function handleBatchFile(event, kind) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  try {
    const text = await file.text();
    await loadBatchSourceFromText({
      kind,
      content: text,
      filename: file.name
    });
  } catch (error) {
    window.alert(`批量导入失败：${error.message}`);
  } finally {
    event.target.value = "";
  }
}

async function loadBatchSourceFromText({ kind, content, filename }) {
  try {
    const payload = await api("/api/batch/parse", {
      method: "POST",
      body: JSON.stringify({
        content,
        filename
      })
    });
    const parsed = {
      items: payload.items || [],
      format: payload.format || "",
      source_name: payload.source_name || filename || "",
      warnings: payload.warnings || []
    };
    if (kind === "match") {
      state.batchMatchSource = parsed;
      state.batchMatchResult = null;
      state.optimizationResult = null;
      renderBatchMatchSource();
      renderBatchMatchOutput();
      renderOptimizationOutput();
      return;
    }
    state.batchGenerateSource = parsed;
    state.generatedBatch = null;
    state.optimizationResult = null;
    renderBatchGenerateSource();
    renderBatchGenerateOutput();
    renderOptimizationOutput();
  } catch (error) {
    window.alert(`批量输入解析失败：${error.message}`);
  }
}

function clearBatchMatchState() {
  state.batchMatchSource = createEmptyBatchSource();
  state.batchMatchResult = null;
  state.optimizationResult = null;
  els.batchMatchInput.value = "";
  renderBatchMatchSource();
  renderBatchMatchOutput();
  renderOptimizationOutput();
}

function clearBatchGeneratorState() {
  state.batchGenerateSource = createEmptyBatchSource();
  state.generatedBatch = null;
  state.optimizationResult = null;
  els.batchGeneratorInput.value = "";
  renderBatchGenerateSource();
  renderBatchGenerateOutput();
  renderOptimizationOutput();
}

function getCurrentTemplate() {
  return state.config?.templates?.find((template) => template.template_id === state.selectedTemplateId) || null;
}

function createBlankTemplate() {
  return {
    template_id: `template.generated.${Date.now()}`,
    query_mode: "metric_query",
    description: "",
    utterances: [],
    required_slots: ["query_operator"],
    optional_slots: [],
    must_terms: [],
    negative_terms: ["原因", "根因", "报告", "总结", "预测"],
    slot_constraints: {},
    slot_extractors: {},
    slot_defaults: {},
    derived_slots: [],
    slot_validations: {},
    llm_slot_extraction: {
      enabled: false,
      slots: [],
      instructions: ""
    },
    metadata: {}
  };
}

function createEmptyBatchSource() {
  return {
    items: [],
    format: "",
    source_name: "",
    warnings: []
  };
}

function getOptimizationItems() {
  return els.optimizeSource.value === "batch-generate" ? state.batchGenerateSource.items : state.batchMatchSource.items;
}

function insertTemplateWithUniqueId(template, options = {}) {
  const next = structuredClone(template);
  const existingIds = new Set(state.config.templates.map((item) => item.template_id));
  const baseId = next.template_id || `template.generated.${Date.now()}`;
  let candidate = baseId;
  let counter = 1;
  while (existingIds.has(candidate)) {
    candidate = `${baseId}.${counter}`;
    counter += 1;
  }
  next.template_id = candidate;
  state.config.templates.unshift(next);
  if (options.select) {
    state.selectedTemplateId = next.template_id;
  }
  return next;
}

function openInspector(title, content) {
  state.inspector = {
    open: true,
    title,
    content
  };
  renderInspector();
}

function closeInspector() {
  state.inspector = {
    open: false,
    title: "",
    content: ""
  };
  renderInspector();
}

function refreshInspectorIfShowingCurrentTemplate() {
  if (!state.inspector.open) {
    return;
  }
  const template = getCurrentTemplate();
  if (!template) {
    return;
  }
  if (state.inspector.title.startsWith("模板详情：")) {
    state.inspector.title = `模板详情：${template.template_id}`;
    state.inspector.content = formatJson(template);
    renderInspector();
  }
}

function renderBatchSourceMeta({ element, source, emptyMessage }) {
  if (!source.items.length) {
    element.className = "muted";
    element.textContent = emptyMessage;
    return;
  }
  element.className = "meta-panel";
  element.innerHTML = `
    <div class="pill-row">
      <span class="pill">items: ${escapeHtml(source.items.length)}</span>
      <span class="pill">format: ${escapeHtml(source.format || "-")}</span>
      <span class="pill">source: ${escapeHtml(source.source_name || "inline")}</span>
    </div>
    ${
      source.warnings.length
        ? `<ul class="hint-list">${source.warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join("")}</ul>`
        : `<p class="muted">已加载批量输入，可直接运行。</p>`
    }
  `;
}

function formatExpectedLabel(item) {
  const parts = [];
  if (item.expected_template_id) {
    parts.push(item.expected_template_id);
  }
  if (item.expected_status) {
    parts.push(item.expected_status);
  }
  if (item.expected_not_full_match) {
    parts.push("status != matched");
  }
  return parts.join(" / ") || "-";
}

function formatMetric(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  return Number(value).toFixed(3);
}

function renderRationaleList(items) {
  const list = Array.isArray(items) ? items.filter(Boolean) : [];
  if (!list.length) {
    return "";
  }
  return `<ul class="hint-list">${list.slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function splitLines(value) {
  return String(value || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function splitCommaList(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function parseMustTerms(value) {
  return splitLines(value).map((line) =>
    line
      .split("|")
      .map((item) => item.trim())
      .filter(Boolean)
  );
}

function formatMustTerms(value) {
  return (value || []).map((group) => group.join(" | ")).join("\n");
}

function formatJson(value) {
  return JSON.stringify(value ?? {}, null, 2);
}

function hydrateLlmSettings() {
  state.llmSettings = {
    apiKey: localStorage.getItem("templateStudio.apiKey") || "",
    baseUrl: localStorage.getItem("templateStudio.baseUrl") || state.llmSettings.baseUrl,
    model: localStorage.getItem("templateStudio.model") || state.llmSettings.model,
    insecureSSL: readBooleanSetting("templateStudio.insecureSSL", state.llmSettings.insecureSSL)
  };
}

function syncLlmSettingsFromInputs() {
  state.llmSettings = {
    apiKey: els.llmApiKey.value.trim(),
    baseUrl: els.llmBaseUrl.value.trim(),
    model: els.llmModel.value.trim(),
    insecureSSL: Boolean(els.llmInsecureSsl.checked)
  };
  persistLlmSettings();
}

function persistLlmSettings() {
  state.llmSettings = {
    apiKey: els.llmApiKey.value.trim(),
    baseUrl: els.llmBaseUrl.value.trim(),
    model: els.llmModel.value.trim(),
    insecureSSL: Boolean(els.llmInsecureSsl.checked)
  };
  localStorage.setItem("templateStudio.apiKey", state.llmSettings.apiKey);
  localStorage.setItem("templateStudio.baseUrl", state.llmSettings.baseUrl);
  localStorage.setItem("templateStudio.model", state.llmSettings.model);
  localStorage.setItem("templateStudio.insecureSSL", state.llmSettings.insecureSSL ? "1" : "0");
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json"
    },
    ...options
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readBooleanSetting(key, fallback = false) {
  const raw = localStorage.getItem(key);
  if (raw == null) {
    return Boolean(fallback);
  }
  return raw === "1" || raw === "true";
}
