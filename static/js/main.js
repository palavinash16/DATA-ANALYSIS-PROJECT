document.addEventListener('DOMContentLoaded', () => {
    // 1. Tab Navigation Logic
    const navBtns = document.querySelectorAll('.nav-btn');
    const tabPages = document.querySelectorAll('.tab-page');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            navBtns.forEach(b => b.classList.remove('active'));
            tabPages.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // 2. Fetch Dashboard Data
    fetchDashboardData();

    // Range Slider Value updates
    const discountSlider = document.getElementById('inp-discount');
    const discountVal = document.getElementById('val-discount');
    if (discountSlider) {
        discountSlider.addEventListener('input', (e) => {
            discountVal.textContent = Math.round(e.target.value * 100) + '%';
        });
    }

    // Predict Loss Button Click
    const btnPredict = document.getElementById('btn-predict');
    if (btnPredict) {
        btnPredict.addEventListener('click', runLossPrediction);
    }
});

let globalDashboardData = null;
let forecastingChart = null;

async function fetchDashboardData() {
    try {
        const res = await fetch('/api/dashboard_data');
        const data = await res.json();
        globalDashboardData = data;

        // Populate KPIs
        document.getElementById('kpi-sales').textContent = '$' + data.kpis.total_sales.toLocaleString();
        document.getElementById('kpi-profit').textContent = '$' + data.kpis.total_profit.toLocaleString();
        document.getElementById('kpi-customers').textContent = data.kpis.total_customers.toLocaleString();
        document.getElementById('kpi-margin').textContent = `Margin: ${data.kpis.profit_margin}%`;
        document.getElementById('kpi-loss-pct').textContent = `${data.kpis.loss_orders_pct}%`;

        // Render Charts
        renderCategoryEdaChart(data.eda.category);
        renderRegionPieChart(data.eda.region);
        renderSubCatBarChart(data.eda.sub_category);
        renderForecastingChart(data.forecasting, 'Gradient Boosting');
        populateMetricsTable(data.forecasting.metrics);
        renderRFMTab(data.rfm);
        renderFeatureImportanceChart(data.feature_importance);

        // Bind forecasting selector
        const selector = document.getElementById('forecasting-model-select');
        if (selector) {
            selector.addEventListener('change', (e) => {
                renderForecastingChart(data.forecasting, e.target.value);
            });
        }
    } catch (err) {
        console.error('Error fetching dashboard data:', err);
    }
}

// Render Revenue & Profit by Category Chart
function renderCategoryEdaChart(categoryData) {
    const categories = categoryData.map(c => c.Category);
    const sales = categoryData.map(c => Math.round(c.Sales));
    const profits = categoryData.map(c => Math.round(c.Profit));

    const options = {
        series: [{ name: 'Sales ($)', data: sales }, { name: 'Profit ($)', data: profits }],
        chart: { type: 'bar', height: 280, background: 'transparent', toolbar: { show: false } },
        theme: { mode: 'dark' },
        colors: ['#3b82f6', '#10b981'],
        plotOptions: { bar: { columnWidth: '45%', borderRadius: 6 } },
        xaxis: { categories: categories, labels: { style: { colors: '#94a3b8' } } },
        yaxis: { labels: { style: { colors: '#94a3b8' } } },
        grid: { borderColor: 'rgba(255,255,255,0.05)' }
    };
    new ApexCharts(document.querySelector("#chart-category-eda"), options).render();
}

// Render Region Pie Chart
function renderRegionPieChart(regionData) {
    const labels = regionData.map(r => r.Region);
    const series = regionData.map(r => Math.round(r.Sales));

    const options = {
        series: series,
        labels: labels,
        chart: { type: 'donut', height: 280, background: 'transparent' },
        theme: { mode: 'dark' },
        colors: ['#3b82f6', '#8b5cf6', '#06b6d4', '#10b981'],
        legend: { position: 'bottom', labels: { colors: '#94a3b8' } },
        dataLabels: { enabled: true }
    };
    new ApexCharts(document.querySelector("#chart-region-pie"), options).render();
}

// Subcategory Bar Chart
function renderSubCatBarChart(subCatData) {
    const subcats = subCatData.map(s => s['Sub-Category']);
    const sales = subCatData.map(s => Math.round(s.Sales));
    const profit = subCatData.map(s => Math.round(s.Profit));

    const options = {
        series: [{ name: 'Sales ($)', data: sales }, { name: 'Profit ($)', data: profit }],
        chart: { type: 'bar', height: 350, background: 'transparent', toolbar: { show: false } },
        theme: { mode: 'dark' },
        colors: ['#06b6d4', '#ef4444'],
        xaxis: { categories: subcats, labels: { style: { colors: '#94a3b8' } } },
        yaxis: { labels: { style: { colors: '#94a3b8' } } },
        grid: { borderColor: 'rgba(255,255,255,0.05)' }
    };
    new ApexCharts(document.querySelector("#chart-subcat-bar"), options).render();
}

// Forecasting Chart
function renderForecastingChart(forecastData, selectedModel) {
    const dates = forecastData.dates;
    const actual = forecastData.actual;
    const predicted = forecastData.predictions[selectedModel] || actual;

    const options = {
        series: [
            { name: 'Actual Sales', data: actual },
            { name: `${selectedModel} Forecast`, data: predicted }
        ],
        chart: { type: 'line', height: 350, background: 'transparent', toolbar: { show: false } },
        theme: { mode: 'dark' },
        colors: ['#f8fafc', '#3b82f6'],
        stroke: { width: [3, 3], curve: 'smooth', dashArray: [0, 5] },
        xaxis: { categories: dates, labels: { style: { colors: '#94a3b8' } } },
        yaxis: { labels: { style: { colors: '#94a3b8' } } },
        grid: { borderColor: 'rgba(255,255,255,0.05)' },
        markers: { size: 4 }
    };

    if (forecastingChart) forecastingChart.destroy();
    forecastingChart = new ApexCharts(document.querySelector("#chart-forecasting"), options);
    forecastingChart.render();
}

// Metrics Leaderboard Table
function populateMetricsTable(metrics) {
    const tbody = document.getElementById('metrics-table-body');
    tbody.innerHTML = metrics.map(m => `
        <tr>
            <td><strong>${m.model}</strong></td>
            <td>$${m.mae.toLocaleString()}</td>
            <td>$${m.rmse.toLocaleString()}</td>
            <td><span class="range-val">${m.r2}</span></td>
            <td>${m.r2 > 0.4 ? '🥇 Excellent' : (m.r2 > 0.3 ? '🥈 Good' : '🥉 Baseline')}</td>
        </tr>
    `).join('');
}

// RFM Tab Render
function renderRFMTab(rfmData) {
    const container = document.getElementById('personas-cards-container');
    container.innerHTML = rfmData.summary.map(s => `
        <div class="glass-card persona-card">
            <h3 class="persona-title">${s.Persona}</h3>
            <span class="kpi-label">Customer Count</span>
            <div class="persona-stat">${s.Customer_Count}</div>
            <p style="font-size: 13px; color: #94a3b8;">Avg Spend: <strong>$${Math.round(s.Monetary).toLocaleString()}</strong></p>
            <p style="font-size: 13px; color: #94a3b8;">Recency: <strong>${Math.round(s.Recency)} days</strong> | Freq: <strong>${s.Frequency.toFixed(1)} orders</strong></p>
        </div>
    `).join('');

    // Scatter Plot
    const points = rfmData.scatter.map(p => ({
        x: Math.round(p.Recency),
        y: Math.round(p.Monetary)
    }));

    const options = {
        series: [{ name: 'Customers (Recency vs Spend)', data: points }],
        chart: { type: 'scatter', height: 350, background: 'transparent' },
        theme: { mode: 'dark' },
        colors: ['#06b6d4'],
        xaxis: { title: { text: 'Recency (Days Inactive)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
        yaxis: { title: { text: 'Monetary Spend ($)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
        grid: { borderColor: 'rgba(255,255,255,0.05)' }
    };
    new ApexCharts(document.querySelector("#chart-rfm-scatter"), options).render();
}

// Feature Importance Chart
function renderFeatureImportanceChart(fiData) {
    const feats = fiData.map(f => f.feature);
    const imps = fiData.map(f => Number((f.importance * 100).toFixed(2)));

    const options = {
        series: [{ name: 'Importance (%)', data: imps }],
        chart: { type: 'bar', height: 300, background: 'transparent', toolbar: { show: false } },
        plotOptions: { bar: { horizontal: true, borderRadius: 6 } },
        theme: { mode: 'dark' },
        colors: ['#8b5cf6'],
        xaxis: { categories: feats, labels: { style: { colors: '#94a3b8' } } },
        yaxis: { labels: { style: { colors: '#94a3b8' } } },
        grid: { borderColor: 'rgba(255,255,255,0.05)' }
    };
    new ApexCharts(document.querySelector("#chart-feature-importance"), options).render();
}

// Run Order Loss Prediction API
async function runLossPrediction() {
    const payload = {
        category: document.getElementById('inp-category').value,
        sub_category: document.getElementById('inp-subcategory').value,
        discount: parseFloat(document.getElementById('inp-discount').value),
        sales: parseFloat(document.getElementById('inp-sales').value),
        quantity: parseInt(document.getElementById('inp-quantity').value),
        region: document.getElementById('inp-region').value,
        segment: document.getElementById('inp-segment').value
    };

    try {
        const res = await fetch('/api/predict_loss', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await res.json();

        if (result.status === 'success') {
            const probEl = document.getElementById('res-prob');
            const badgeEl = document.getElementById('res-badge');
            const recEl = document.getElementById('res-rec');

            probEl.textContent = `${result.loss_probability}%`;
            badgeEl.textContent = result.risk_level;
            recEl.textContent = result.recommendation;

            if (result.is_loss) {
                probEl.style.color = '#ef4444';
                badgeEl.style.background = 'rgba(239, 68, 68, 0.2)';
                badgeEl.style.color = '#ef4444';
            } else {
                probEl.style.color = '#10b981';
                badgeEl.style.background = 'rgba(16, 185, 129, 0.2)';
                badgeEl.style.color = '#10b981';
            }
        }
    } catch (err) {
        console.error('Prediction error:', err);
    }
}
