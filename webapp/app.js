/* ═══ Movie Success Prediction — Frontend Logic ═══ */

// ─── Slider live values ───
document.getElementById("cast_popularity").addEventListener("input", (e) => {
    document.getElementById("cast_val").textContent = e.target.value;
});
document.getElementById("director_score").addEventListener("input", (e) => {
    document.getElementById("dir_val").textContent = e.target.value;
});

// ─── Budget formatting hint ───
document.getElementById("budget").addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    const hint = document.getElementById("budget-hint");
    if (!val || val <= 0) { hint.textContent = "Enter production budget"; return; }
    hint.textContent = val >= 1e9 ? `$${(val/1e9).toFixed(1)}B`
                     : val >= 1e6 ? `$${(val/1e6).toFixed(1)}M`
                     : val >= 1e3 ? `$${(val/1e3).toFixed(0)}K`
                     : `$${val.toLocaleString()}`;
});

// ─── Navigation ───
function showPage(page) {
    document.querySelectorAll(".page").forEach((p) => p.classList.remove("active"));
    document.querySelectorAll(".nav-tab").forEach((t) => t.classList.remove("active"));
    document.getElementById(`page-${page}`).classList.add("active");
    document.getElementById(`tab-${page}`).classList.add("active");
}

document.getElementById("tab-predict").addEventListener("click", () => showPage("predict"));
document.getElementById("tab-results").addEventListener("click", () => {
    if (!document.getElementById("tab-results").disabled) showPage("results");
});
document.getElementById("back-btn").addEventListener("click", () => showPage("predict"));

// ─── Format helpers ───
function fmt$(n) {
    if (Math.abs(n) >= 1e9) return "$" + (n / 1e9).toFixed(2) + "B";
    if (Math.abs(n) >= 1e6) return "$" + (n / 1e6).toFixed(1) + "M";
    if (Math.abs(n) >= 1e3) return "$" + (n / 1e3).toFixed(0) + "K";
    return "$" + n.toLocaleString();
}

// ─── Animated counter ───
function animateCounter(el, endVal, formatter, duration = 1200) {
    const start = performance.now();
    function tick(now) {
        const t = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - t, 3);
        el.textContent = formatter(eased * endVal);
        if (t < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
}

// ─── Chart instances (for cleanup) ───
let chartBudgetRev = null;
let chartBreakdown = null;

// ─── Form submission ───
document.getElementById("predict-form").addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = document.getElementById("predict-btn");
    const btnText = btn.querySelector(".btn-text");
    const btnLoading = btn.querySelector(".btn-loading");
    btnText.style.display = "none";
    btnLoading.style.display = "inline";
    btn.disabled = true;

    const payload = {
        movie_title: document.getElementById("movie_title").value,
        budget: document.getElementById("budget").value,
        genre: document.getElementById("genre").value,
        runtime: document.getElementById("runtime").value,
        release_month: document.getElementById("release_month").value,
        imdb_rating: document.getElementById("imdb_rating").value,
        cast_popularity: document.getElementById("cast_popularity").value,
        director_score: document.getElementById("director_score").value,
    };

    try {
        const resp = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();

        if (!data.success) {
            alert("Prediction failed: " + (data.error || "Unknown error"));
            return;
        }

        renderResults(payload, data);
    } catch (err) {
        alert("Network error: " + err.message);
    } finally {
        btnText.style.display = "inline";
        btnLoading.style.display = "none";
        btn.disabled = false;
    }
});

// ─── Render results ───
function renderResults(input, data) {
    // Enable results tab & switch
    document.getElementById("tab-results").disabled = false;
    showPage("results");

    // Title
    document.getElementById("results-movie-title").textContent =
        `Results for "${input.movie_title || "Untitled Movie"}"`;

    // Verdict banner
    const banner = document.getElementById("verdict-banner");
    banner.className = "verdict-banner " + data.verdict_class;

    const emojiMap = { blockbuster: "🏆", hit: "🎉", average: "🎬", flop: "💔" };
    const colorMap = { blockbuster: "#f59e0b", hit: "#10b981", average: "#eab308", flop: "#ef4444" };

    document.getElementById("verdict-emoji").textContent = emojiMap[data.verdict_class] || "🎬";
    document.getElementById("verdict-text").textContent = data.verdict;
    document.getElementById("verdict-text").style.color = colorMap[data.verdict_class];
    document.getElementById("verdict-confidence").textContent = data.confidence;

    // KPI cards with animation
    animateCounter(document.getElementById("res-revenue"), data.predicted_revenue, fmt$, 1400);
    animateCounter(document.getElementById("res-budget"), data.budget, fmt$, 1000);
    animateCounter(document.getElementById("res-profit"), data.profit,
        (v) => (v < 0 ? "-" : "") + fmt$(Math.abs(v)), 1400);

    const roi = ((data.predicted_revenue - data.budget) / Math.max(data.budget, 1)) * 100;
    animateCounter(document.getElementById("res-roi"), roi, (v) => v.toFixed(1) + "%", 1200);

    // Profit card color
    const profitCard = document.getElementById("card-profit");
    profitCard.classList.toggle("negative", data.profit < 0);

    // Model R²
    document.getElementById("model-r2").textContent = data.model_r2;

    // Charts
    renderBudgetRevenueChart(data);
    renderBreakdownChart(data);

    // Feature impacts
    renderImpacts(data.feature_impacts);
}

// ─── Budget vs Revenue Bar Chart ───
function renderBudgetRevenueChart(data) {
    if (chartBudgetRev) chartBudgetRev.destroy();
    const ctx = document.getElementById("chart-budget-rev").getContext("2d");

    chartBudgetRev = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["Budget", "Predicted Revenue"],
            datasets: [{
                data: [data.budget / 1e6, data.predicted_revenue / 1e6],
                backgroundColor: [
                    "rgba(59, 130, 246, 0.7)",
                    data.predicted_revenue >= data.budget
                        ? "rgba(16, 185, 129, 0.7)"
                        : "rgba(239, 68, 68, 0.7)",
                ],
                borderColor: [
                    "#3b82f6",
                    data.predicted_revenue >= data.budget ? "#10b981" : "#ef4444",
                ],
                borderWidth: 2,
                borderRadius: 10,
                borderSkipped: false,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: { label: (ctx) => `$${ctx.raw.toFixed(1)}M` },
                },
            },
            scales: {
                x: { ticks: { color: "#64748b", font: { family: "Inter" } }, grid: { display: false } },
                y: {
                    ticks: {
                        color: "#64748b",
                        font: { family: "Inter" },
                        callback: (v) => "$" + v + "M",
                    },
                    grid: { color: "rgba(255,255,255,0.04)" },
                },
            },
            animation: { duration: 1200, easing: "easeOutQuart" },
        },
    });
}

// ─── Breakdown Doughnut Chart ───
function renderBreakdownChart(data) {
    if (chartBreakdown) chartBreakdown.destroy();
    const ctx = document.getElementById("chart-breakdown").getContext("2d");

    const profit = Math.max(0, data.profit);
    const loss = Math.max(0, -data.profit);

    chartBreakdown = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: profit > 0
                ? ["Budget", "Profit"]
                : ["Budget", "Loss"],
            datasets: [{
                data: profit > 0
                    ? [data.budget / 1e6, profit / 1e6]
                    : [data.predicted_revenue / 1e6, loss / 1e6],
                backgroundColor: profit > 0
                    ? ["rgba(59, 130, 246, 0.7)", "rgba(16, 185, 129, 0.7)"]
                    : ["rgba(59, 130, 246, 0.7)", "rgba(239, 68, 68, 0.7)"],
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "65%",
            plugins: {
                legend: {
                    position: "bottom",
                    labels: { color: "#94a3b8", font: { family: "Inter", size: 12 }, padding: 16 },
                },
                tooltip: {
                    callbacks: { label: (ctx) => `${ctx.label}: $${ctx.raw.toFixed(1)}M` },
                },
            },
            animation: { animateRotate: true, duration: 1400 },
        },
    });
}

// ─── Feature Impact Cards ───
function renderImpacts(impacts) {
    const grid = document.getElementById("impact-grid");
    if (!impacts || impacts.length === 0) {
        grid.innerHTML = '<p style="color: var(--text-muted);">No specific factors to highlight.</p>';
        return;
    }
    grid.innerHTML = impacts.map((imp) => `
        <div class="impact-item ${imp.impact}">
            <div class="impact-arrow">${imp.impact === "positive" ? "🟢" : "🔴"}</div>
            <div class="impact-content">
                <div class="impact-feature">${imp.feature}</div>
                <div class="impact-detail">${imp.detail}</div>
            </div>
        </div>
    `).join("");
}
