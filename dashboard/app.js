/* ─── Movie Box Office Dashboard — app.js ─── */

const COLORS = {
    indigo:  "#6366f1",
    purple:  "#a855f7",
    pink:    "#ec4899",
    teal:    "#14b8a6",
    amber:   "#f59e0b",
    blue:    "#3b82f6",
    red:     "#ef4444",
    green:   "#10b981",
};
const PALETTE = Object.values(COLORS);

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: { labels: { color: "#94a3b8", font: { family: "Inter", size: 12 } } },
    },
    scales: {
        x: { ticks: { color: "#64748b", font: { family: "Inter" } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#64748b", font: { family: "Inter" } }, grid: { color: "rgba(255,255,255,0.04)" } },
    },
};

function fmt$(n) {
    if (n >= 1e9) return "$" + (n / 1e9).toFixed(1) + "B";
    if (n >= 1e6) return "$" + (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return "$" + (n / 1e3).toFixed(1) + "K";
    return "$" + n.toFixed(0);
}

/* ─── Load data ─── */
async function loadData() {
    const resp = await fetch("../outputs/dashboard_data.json");
    return resp.json();
}

/* ─── Populate KPIs ─── */
function renderKPIs(data) {
    const best = data.best_model;
    animateValue("kpi-r2-val", 0, best.r2, 1200, (v) => v.toFixed(4));
    animateValue("kpi-mae-val", 0, best.mae / 1e6, 1200, (v) => "$" + v.toFixed(1) + "M");
    document.getElementById("kpi-best-model").textContent = best.name;
    document.getElementById("kpi-models-val").textContent = data.model_results.length;
    document.getElementById("kpi-features-val").textContent = data.dataset_stats.features_used;
    document.getElementById("kpi-rows-val").textContent = data.dataset_stats.total_rows.toLocaleString() + " movies";
}

function animateValue(id, start, end, duration, formatter) {
    const el = document.getElementById(id);
    const startTime = performance.now();
    function tick(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = start + (end - start) * eased;
        el.textContent = formatter(current);
        if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

/* ─── R² Bar Chart ─── */
function renderR2Chart(data) {
    const sorted = [...data.model_results].sort((a, b) => b.R2 - a.R2);
    new Chart(document.getElementById("chart-r2"), {
        type: "bar",
        data: {
            labels: sorted.map((m) => m.Model),
            datasets: [{
                label: "R² Score",
                data: sorted.map((m) => m.R2),
                backgroundColor: PALETTE.slice(0, sorted.length),
                borderRadius: 8,
                borderSkipped: false,
            }],
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: { ...chartDefaults.plugins, legend: { display: false } },
            scales: {
                x: { ...chartDefaults.scales.x, min: 0, max: 1 },
                y: { ...chartDefaults.scales.y },
            },
        },
    });
}

/* ─── RMSE Bar Chart ─── */
function renderRMSEChart(data) {
    const sorted = [...data.model_results].sort((a, b) => a.RMSE - b.RMSE);
    new Chart(document.getElementById("chart-rmse"), {
        type: "bar",
        data: {
            labels: sorted.map((m) => m.Model),
            datasets: [{
                label: "RMSE ($M)",
                data: sorted.map((m) => m.RMSE / 1e6),
                backgroundColor: PALETTE.slice(0, sorted.length),
                borderRadius: 8,
                borderSkipped: false,
            }],
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: { ...chartDefaults.plugins, legend: { display: false } },
        },
    });
}

/* ─── Cross-Validation Chart ─── */
function renderCVChart(data) {
    const models = data.model_results.filter((m) => m.CV_Std_R2 > 0);
    const sorted = [...models].sort((a, b) => b.CV_Mean_R2 - a.CV_Mean_R2);
    new Chart(document.getElementById("chart-cv"), {
        type: "bar",
        data: {
            labels: sorted.map((m) => m.Model),
            datasets: [{
                label: "CV Mean R²",
                data: sorted.map((m) => m.CV_Mean_R2),
                backgroundColor: PALETTE.slice(0, sorted.length).map((c) => c + "CC"),
                borderColor: PALETTE.slice(0, sorted.length),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }],
        },
        options: {
            ...chartDefaults,
            plugins: {
                ...chartDefaults.plugins,
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        afterLabel: (ctx) => {
                            const m = sorted[ctx.dataIndex];
                            return `Std: ±${m.CV_Std_R2.toFixed(4)}`;
                        },
                    },
                },
            },
            scales: {
                ...chartDefaults.scales,
                y: { ...chartDefaults.scales.y, min: 0, max: 1, title: { display: true, text: "R² Score", color: "#64748b" } },
            },
        },
    });
}

/* ─── Scatter: Actual vs Predicted ─── */
function renderScatterChart(data) {
    const actual = data.predictions_sample.actual;
    const predicted = data.predictions_sample.predicted;
    const points = actual.map((a, i) => ({ x: a / 1e6, y: predicted[i] / 1e6 }));
    const maxVal = Math.max(...actual, ...predicted) / 1e6 * 1.1;

    new Chart(document.getElementById("chart-scatter"), {
        type: "scatter",
        data: {
            datasets: [
                {
                    label: "Predictions",
                    data: points,
                    backgroundColor: COLORS.indigo + "99",
                    pointRadius: 5,
                    pointHoverRadius: 8,
                },
                {
                    label: "Perfect",
                    data: [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }],
                    type: "line",
                    borderColor: COLORS.red,
                    borderDash: [8, 4],
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                },
            ],
        },
        options: {
            ...chartDefaults,
            scales: {
                x: { ...chartDefaults.scales.x, title: { display: true, text: "Actual ($M)", color: "#64748b" } },
                y: { ...chartDefaults.scales.y, title: { display: true, text: "Predicted ($M)", color: "#64748b" } },
            },
        },
    });
}

/* ─── Residuals Histogram ─── */
function renderResidualChart(data) {
    const residuals = data.residuals_sample.map((r) => r / 1e6);
    const min = Math.min(...residuals);
    const max = Math.max(...residuals);
    const binCount = 20;
    const binWidth = (max - min) / binCount;
    const bins = Array(binCount).fill(0);
    const labels = [];

    for (let i = 0; i < binCount; i++) {
        const lo = min + i * binWidth;
        labels.push(lo.toFixed(0));
    }
    residuals.forEach((r) => {
        let idx = Math.floor((r - min) / binWidth);
        if (idx >= binCount) idx = binCount - 1;
        bins[idx]++;
    });

    new Chart(document.getElementById("chart-residuals"), {
        type: "bar",
        data: {
            labels,
            datasets: [{
                label: "Frequency",
                data: bins,
                backgroundColor: COLORS.pink + "AA",
                borderColor: COLORS.pink,
                borderWidth: 1,
                borderRadius: 4,
            }],
        },
        options: {
            ...chartDefaults,
            plugins: { ...chartDefaults.plugins, legend: { display: false } },
            scales: {
                x: { ...chartDefaults.scales.x, title: { display: true, text: "Residual ($M)", color: "#64748b" } },
                y: { ...chartDefaults.scales.y, title: { display: true, text: "Count", color: "#64748b" } },
            },
        },
    });
}

/* ─── Feature Importance ─── */
function renderFeatureChart(data) {
    const feats = [...data.feature_importance].sort((a, b) => a.Importance - b.Importance);
    new Chart(document.getElementById("chart-features"), {
        type: "bar",
        data: {
            labels: feats.map((f) => f.Feature),
            datasets: [{
                label: "Importance",
                data: feats.map((f) => f.Importance),
                backgroundColor: feats.map((_, i) => {
                    const t = i / (feats.length - 1);
                    return `hsl(${160 + t * 80}, 70%, ${45 + t * 15}%)`;
                }),
                borderRadius: 6,
                borderSkipped: false,
            }],
        },
        options: {
            ...chartDefaults,
            indexAxis: "y",
            plugins: { ...chartDefaults.plugins, legend: { display: false } },
        },
    });
}

/* ─── Results Table ─── */
function renderTable(data) {
    const tbody = document.getElementById("results-tbody");
    const sorted = [...data.model_results].sort((a, b) => b.R2 - a.R2);
    tbody.innerHTML = sorted.map((m, i) => {
        const rank = i + 1;
        const rankClass = rank <= 3 ? `rank-${rank}` : "rank-other";
        const cvStr = m.CV_Std_R2 > 0
            ? `${m.CV_Mean_R2.toFixed(4)} ± ${m.CV_Std_R2.toFixed(4)}`
            : `${m.CV_Mean_R2.toFixed(4)}`;
        return `<tr>
            <td><span class="rank-badge ${rankClass}">${rank}</span></td>
            <td>${m.Model}</td>
            <td>${m.R2.toFixed(4)}</td>
            <td>${cvStr}</td>
            <td>${(m.MAE / 1e6).toFixed(1)}</td>
            <td>${(m.RMSE / 1e6).toFixed(1)}</td>
        </tr>`;
    }).join("");
}

/* ─── Params Grid ─── */
function renderParams(data) {
    const grid = document.getElementById("params-grid");
    const params = data.best_model.best_params;
    grid.innerHTML = Object.entries(params).map(([k, v]) =>
        `<div class="param-card">
            <div class="param-key">${k}</div>
            <div class="param-val">${v}</div>
        </div>`
    ).join("");
}

/* ─── Init ─── */
async function init() {
    const data = await loadData();
    renderKPIs(data);
    renderR2Chart(data);
    renderRMSEChart(data);
    renderCVChart(data);
    renderScatterChart(data);
    renderResidualChart(data);
    renderFeatureChart(data);
    renderTable(data);
    renderParams(data);
}

init();
