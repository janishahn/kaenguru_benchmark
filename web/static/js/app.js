(function(){
  function $(selector, root){
    return (root || document).querySelector(selector);
  }

  function $$(selector, root){
    return Array.from((root || document).querySelectorAll(selector));
  }

  function fetchJSON(url, params){
    if (params && Object.keys(params).length){
      var search = new URLSearchParams();
      Object.keys(params).forEach(function(key){
        var value = params[key];
        if (value === undefined || value === null || value === '') return;
        if (Array.isArray(value)){
          value.forEach(function(v){ search.append(key, v); });
        } else {
          search.append(key, value);
        }
      });
      if (url.indexOf('?') === -1){
        url += '?' + search.toString();
      } else {
        url += '&' + search.toString();
      }
    }
    return fetch(url, { headers: { 'Accept': 'application/json' } })
      .then(function(response){
        if (!response.ok){
          throw new Error('Request failed: ' + response.status);
        }
        return response.json();
      });
  }

  function saveTheme(theme){
    try {
      window.localStorage.setItem('dashboard:theme', theme);
    } catch (err){ /* ignore */ }
  }

  function loadTheme(){
    try {
      return window.localStorage.getItem('dashboard:theme');
    } catch (err){
      return null;
    }
  }

  function applyTheme(theme){
    var root = document.documentElement;
    if (theme === 'dark'){
      root.setAttribute('data-theme', 'dark');
    } else {
      root.setAttribute('data-theme', 'light');
    }
    saveTheme(theme);
  }

  var overviewListenersBound = false;

  function initThemeToggle(){
    var saved = loadTheme();
    if (saved){
      applyTheme(saved);
    }
    var toggle = $('#theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', function(){
      var current = document.documentElement.getAttribute('data-theme');
      var next = current === 'dark' ? 'light' : 'dark';
      applyTheme(next);
      document.dispatchEvent(new CustomEvent('dashboard:themechange', { detail: { theme: next } }));
    });
  }

  function initOverview(){
    var root = document.getElementById('overview-root');
    if (!root) return;

    if (!overviewListenersBound){
      document.body.addEventListener('htmx:beforeRequest', function(event){
        var current = document.getElementById('overview-root');
        if (!current) return;
        var source = event.detail && event.detail.elt;
        if (source && current.contains(source)){
          current.classList.add('is-loading');
        }
      });
      document.body.addEventListener('htmx:afterRequest', function(event){
        var current = document.getElementById('overview-root');
        if (!current) return;
        var source = event.detail && event.detail.elt;
        if (source && current.contains(source)){
          current.classList.remove('is-loading');
        }
      });
      document.body.addEventListener('htmx:afterSwap', function(event){
        if (event.target && event.target.id === 'overview-root'){
          event.target.classList.remove('is-loading');
          initOverview();
        }
      });
      overviewListenersBound = true;
    }

    var form = document.getElementById('overview-filters');
    if (!form) return;
    var toggleBtn = form.querySelector('[data-action="toggle-filters"]');
    var storageKey = 'dashboard:overview:filtersCollapsed';

    function setToggleState(collapsed){
      form.classList.toggle('is-collapsed', collapsed);
      if (toggleBtn){
        toggleBtn.textContent = collapsed ? 'Show advanced filters' : 'Hide advanced filters';
        toggleBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      }
    }

    var initial = true;
    try {
      var stored = window.localStorage.getItem(storageKey);
      if (stored !== null){
        initial = stored === '1';
      }
    } catch (err){
      /* ignore */
    }
    setToggleState(initial);

    if (toggleBtn && !toggleBtn.dataset.bound){
      toggleBtn.dataset.bound = 'true';
      toggleBtn.addEventListener('click', function(){
        var next = !form.classList.contains('is-collapsed');
        setToggleState(next);
        try {
          window.localStorage.setItem(storageKey, next ? '1' : '0');
        } catch (err){ /* ignore */ }
      });
    }

    if (!form.dataset.autosubmitBound){
      form.dataset.autosubmitBound = 'true';
      var submitDelay = 250;
      var submitTimer = null;
      var activeController = null;

      function submitWithFallback(){
        var hxGet = form.getAttribute('hx-get');
        var hxTarget = form.getAttribute('hx-target');
        if (!hxGet || !hxTarget || typeof window.fetch !== 'function'){
          return false;
        }

        if (activeController){
          activeController.abort();
        }
        var controller = new AbortController();
        activeController = controller;

        var rootEl = document.getElementById('overview-root');
        if (rootEl){
          rootEl.classList.add('is-loading');
        }

        var data = new FormData(form);
        var params = new URLSearchParams();
        data.forEach(function(value, key){
          params.append(key, value);
        });

        var url = hxGet;
        var query = params.toString();
        if (query){
          url += (url.indexOf('?') === -1 ? '?' : '&') + query;
        }
        var requestUrl = new URL(url, window.location.origin);

        fetch(requestUrl.toString(), {
          credentials: 'same-origin',
          headers: { 'HX-Request': 'true' },
          signal: controller.signal
        })
          .then(function(response){
            if (!response.ok){
              throw new Error('Request failed: ' + response.status);
            }
            return response.text().then(function(body){
              return { body: body, response: response };
            });
          })
          .then(function(payload){
            if (controller.signal.aborted){
              return;
            }
            var parser = new DOMParser();
            var doc = parser.parseFromString(payload.body, 'text/html');
            var replacement = doc.querySelector(hxTarget);
            var targetEl = document.querySelector(hxTarget);
            if (replacement && targetEl){
              targetEl.replaceWith(replacement);
              if (form.getAttribute('hx-push-url') === 'true'){
                var finalUrl = payload.response && payload.response.url ? new URL(payload.response.url) : requestUrl;
                window.history.pushState({}, '', finalUrl.pathname + finalUrl.search + finalUrl.hash);
              }
              var refreshed = document.querySelector('#overview-root');
              if (refreshed){
                refreshed.classList.remove('is-loading');
              }
              initOverview();
            }
          })
          .catch(function(err){
            if (err.name === 'AbortError'){
              return;
            }
            console.error('Overview refresh failed', err);
            if (typeof form.requestSubmit === 'function'){
              form.requestSubmit();
            } else {
              form.submit();
            }
          })
          .finally(function(){
            if (activeController === controller){
              activeController = null;
            }
            if (rootEl){
              rootEl.classList.remove('is-loading');
            }
          });

        return true;
      }

      function queueSubmit(){
        if (submitTimer){
          window.clearTimeout(submitTimer);
        }
        submitTimer = window.setTimeout(function(){
          if (window.htmx && typeof window.htmx.trigger === 'function' && window.htmx.version !== '0.0.1-local'){
            window.htmx.trigger(form, 'submit');
          } else if (submitWithFallback()){
            // handled via fetch fallback
          } else if (typeof form.requestSubmit === 'function'){
            form.requestSubmit();
          } else {
            form.submit();
          }
        }, submitDelay);
      }

      form.addEventListener('change', function(event){
        if (!event.target || !event.target.name) return;
        queueSubmit();
      });

      form.addEventListener('input', function(event){
        var target = event.target;
        if (!target || !target.name) return;
        if (target.matches('input[type="search"], input[type="text"], input[type="date"], input[type="number"], textarea')){
          queueSubmit();
        }
      });
    }
  }

  function initRunDetail(){
    var container = document.querySelector('.run-detail');
    if (!container) return;
    var bootstrap = $('#run-bootstrap');
    if (!bootstrap) return;
    var payload = {};
    try {
      payload = JSON.parse(bootstrap.textContent || '{}');
    } catch (err){
      console.error('Failed to parse run bootstrap', err);
      return;
    }
    var runId = payload.runId;
    if (!runId) return;

    var charts = new window.DashboardCharts();
    var lastAggregates = null;
    var state = {
      runId: runId,
      filters: {
        group: [],
        year: [],
        language: [],
        predicted: [],
        reasoning_mode: [],
        warning_types: [],
        warnings_present: '',
        multimodal: '',
        correctness: '',
        points_min: '',
        points_max: '',
        latency_min: '',
        latency_max: '',
        tokens_min: '',
        tokens_max: '',
        reasoning_tokens_min: '',
        reasoning_tokens_max: '',
        cost_min: '',
        cost_max: '',
        page: 1,
        page_size: 25,
        sort_by: 'id',
        sort_dir: 'asc'
      },
      humanComparison: null
    };

    var tableBody = $('#results-table tbody');
    var pagination = $('#results-pagination');
    var filterForm = $('#filters-form');
    var chipContainers = $$('.chip-select, select[data-facet]', filterForm);
    var sortBy = $('#sort-by');
    var sortDir = $('#sort-dir');
    var pageSize = $('#page-size');
    var resetBtn = $('#reset-filters');
    var exportJson = $('#export-json');
    var exportCsv = $('#export-csv');
    var warningList = $('#warning-toplist');
    var failuresList = $('#failures-timeline');
    var rowDrawer = $('#row-drawer');
    var rowMeta = $('#row-meta');
    var rowRationale = $('#row-rationale');
    var rowRaw = $('#row-raw');
    var rowDataset = $('#row-dataset');
    var rawToggle = $('#toggle-raw-response');
    var rawResponse = $('#raw-response');
    var closeRowBtn = $('#close-row');
    var activeFilters = $('#active-filters');
    var presetName = $('#preset-name');
    var savePresetBtn = $('#save-preset');
    var loadPresetBtn = $('#load-preset');
    var presetSelect = $('#preset-select');
    var subsetSummaryBody = $('#correctness-summary tbody');
    var subsetSummaryNote = $('#subset-summary-note');
    var subsetToggle = $('#subset-correctness-toggle');
    var gradeChart = $('#chart-grade-subset');
    var pointsChart = $('#chart-points-subset');
    var pointsEarnedChart = $('#chart-points-earned');
    var subsetLanguageList = $('#subset-language-list');
    var subsetReasoningList = $('#subset-reasoning-list');
    var subsetStatsList = $('#subset-stats-list');
    var subsetUIEnabled = Boolean(subsetSummaryBody || subsetToggle || gradeChart || pointsChart || pointsEarnedChart || subsetStatsList);
    var subsetMetrics = [];
    var selectedSubsetKey = null;
    var SUBSET_TABLE_ORDER = ['all','incorrect','correct','unknown'];
    var SUBSET_TOGGLE_ORDER = ['incorrect','correct','unknown','all'];
    var humanCards = $('#human-baseline-cards');
    var humanPercentileValue = $('#run-human-percentile');
    var humanPercentileNote = $('#run-human-percentile-note');
    var humanZScoreValue = $('#run-human-zscore');
    var humanZScoreNote = $('#run-human-zscore-note');
    var humanScoreValue = $('#run-human-score');
    var humanScoreNote = $('#run-human-score-note');

    function updatePresetSelect(){
      var presets = loadPresets();
      presetSelect.innerHTML = '';
      Object.keys(presets).sort().forEach(function(name){
        var option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        presetSelect.appendChild(option);
      });
    }

    function loadPresets(){
      try {
        var raw = window.localStorage.getItem('dashboard:presets:' + runId);
        return raw ? JSON.parse(raw) : {};
      } catch (err){
        return {};
      }
    }

    function savePresets(presets){
      try {
        window.localStorage.setItem('dashboard:presets:' + runId, JSON.stringify(presets));
      } catch (err){
        console.warn('Unable to save preset', err);
      }
    }

    function serializeFilters(){
      var params = Object.assign({}, state.filters);
      var arrays = ['group','year','language','predicted','reasoning_mode','warning_types'];
      arrays.forEach(function(key){
        params[key] = state.filters[key].slice();
      });
      return params;
    }

    function restoreFilters(data){
      Object.keys(state.filters).forEach(function(key){
        var value = data[key];
        if (Array.isArray(state.filters[key])){
          state.filters[key] = Array.isArray(value) ? value.slice() : [];
        } else {
          state.filters[key] = value ?? '';
        }
      });
      state.filters.page = 1;
      syncControls();
      refresh();
    }

    function facetToField(facet){
      switch (facet){
        case 'groups': return 'group';
        case 'years': return 'year';
        case 'languages': return 'language';
        case 'predicted_letters': return 'predicted';
        case 'reasoning_modes': return 'reasoning_mode';
        case 'warning_types': return 'warning_types';
        default: return facet;
      }
    }

    function syncControls(){
      // Chip selections
      chipContainers.forEach(function(container){
        var facet = container.dataset.facet;
        var field = facetToField(facet);
        if (container.tagName === 'SELECT'){
          Array.from(container.options).forEach(function(option){
            option.selected = state.filters[field].indexOf(option.value) !== -1;
          });
          return;
        }
        $$('.chip-select button', container).forEach(function(button){
          var value = button.dataset.value;
          var list = state.filters[field];
          if (Array.isArray(list) && list.indexOf(value) !== -1){
            button.classList.add('active');
          } else {
            button.classList.remove('active');
          }
        });
      });
      filterForm.multimodal.value = state.filters.multimodal;
      filterForm.correctness.value = state.filters.correctness;
      filterForm.warnings_present.value = state.filters.warnings_present;
      ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','reasoning_tokens_min','reasoning_tokens_max','cost_min','cost_max'].forEach(function(field){
        filterForm.elements[field].value = state.filters[field];
      });
      sortBy.value = state.filters.sort_by;
      sortDir.value = state.filters.sort_dir;
      pageSize.value = state.filters.page_size;
      renderActiveFilters();
    }

    function renderActiveFilters(){
      if (!activeFilters) return;
      var items = [];
      ['group','year','language','predicted','reasoning_mode','warning_types'].forEach(function(key){
        if (state.filters[key] && state.filters[key].length){
          items.push(key + ': ' + state.filters[key].join(', '));
        }
      });
      ['multimodal','correctness','warnings_present'].forEach(function(key){
        if (state.filters[key]){
          items.push(key + ': ' + state.filters[key]);
        }
      });
      ['points','latency','tokens','reasoning_tokens','cost'].forEach(function(prefix){
        var minKey = prefix + '_min';
        var maxKey = prefix + '_max';
        if (state.filters[minKey] || state.filters[maxKey]){
          items.push(prefix + ': ' + (state.filters[minKey] || '–') + '…' + (state.filters[maxKey] || '–'));
        }
      });
      activeFilters.textContent = items.join(' · ');
    }

    function formatPercent(value){
      if (value === null || value === undefined) return '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return '–';
      return (numeric * 100).toLocaleString(undefined, { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%';
    }

    function formatNumber(value, decimals, fallback){
      if (value === null || value === undefined || value === '') return fallback || '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return fallback || '–';
      if (decimals === undefined){
        return numeric.toLocaleString();
      }
      return numeric.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
    }

    function formatCurrency(value){
      if (value === null || value === undefined || value === '') return '–';
      var numeric = Number(value);
      if (!isFinite(numeric)) return '–';
      return '$' + numeric.toLocaleString(undefined, { minimumFractionDigits: 4, maximumFractionDigits: 4 });
    }

    function describeSummary(summary, decimals, options){
      if (!summary || !summary.count){
        return 'No data';
      }
      var formatter = (options && options.formatter) || function(value){ return formatNumber(value, decimals); };
      var parts = [];
      if (summary.mean !== null && summary.mean !== undefined){
        var meanText = formatter(summary.mean);
        if (meanText && meanText !== '–'){
          parts.push('mean ' + meanText);
        }
      }
      if (summary.median !== null && summary.median !== undefined){
        var medianText = formatter(summary.median);
        if (medianText && medianText !== '–'){
          parts.push('median ' + medianText);
        }
      }
      if (summary.p25 !== null && summary.p25 !== undefined && summary.p75 !== null && summary.p75 !== undefined){
        var low = formatter(summary.p25);
        var high = formatter(summary.p75);
        if (low && high && low !== '–' && high !== '–'){
          parts.push('p25–p75 ' + low + '…' + high);
        }
      }
      if (!parts.length){
        return 'No data';
      }
      return parts.join(' · ');
    }

    function renderDistributionList(element, entries){
      if (!element) return;
      element.innerHTML = '';
      var list = entries || [];
      if (!list.length){
        var empty = document.createElement('li');
        empty.textContent = 'No data available.';
        element.appendChild(empty);
        return;
      }
      list.forEach(function(entry){
        var li = document.createElement('li');
        var name = document.createElement('span');
        name.textContent = entry && entry.label ? entry.label : 'Unknown';
        var value = document.createElement('span');
        var percentText = entry && entry.percentage !== undefined && entry.percentage !== null ? formatPercent(entry.percentage) : '–';
        var countText = entry && entry.count !== undefined && entry.count !== null ? entry.count : 0;
        value.textContent = countText + ' · ' + percentText;
        li.appendChild(name);
        li.appendChild(value);
        element.appendChild(li);
      });
    }

    function createStatItem(label, value){
      var li = document.createElement('li');
      var name = document.createElement('span');
      name.textContent = label;
      var detail = document.createElement('span');
      detail.textContent = value;
      li.appendChild(name);
      li.appendChild(detail);
      return li;
    }

    function renderSubsetStats(metric){
      if (!subsetStatsList) return;
      subsetStatsList.innerHTML = '';
      if (!metric || !metric.count){
        subsetStatsList.appendChild(createStatItem('Subset', 'No data for the current selection.'));
        return;
      }
      subsetStatsList.appendChild(createStatItem('Multimodal share', formatPercent(metric.multimodal_share)));
      subsetStatsList.appendChild(createStatItem('Points', describeSummary(metric.points_summary, 2)));
      if (metric.points_earned_summary && metric.points_earned_summary.count){
        subsetStatsList.appendChild(createStatItem('Points earned', describeSummary(metric.points_earned_summary, 2)));
      } else {
        subsetStatsList.appendChild(createStatItem('Points earned', 'No scoring data recorded.'));
      }
      subsetStatsList.appendChild(createStatItem('Latency', describeSummary(metric.latency_summary, 1, {
        formatter: function(value){ return formatNumber(value, 1) + ' ms'; }
      })));
      subsetStatsList.appendChild(createStatItem('Total tokens', describeSummary(metric.tokens_summary, 0)));
      subsetStatsList.appendChild(createStatItem('Cost', describeSummary(metric.cost_summary, 4, {
        formatter: function(value){ return formatCurrency(value); }
      })));
    }

    function findDefaultSubsetKey(metrics){
      if (!metrics || !metrics.length) return null;
      var order = ['incorrect', 'correct', 'all', 'unknown'];
      for (var i = 0; i < order.length; i++){
        var key = order[i];
        var match = metrics.find(function(item){ return item.key === key && item.count > 0; });
        if (match){
          return key;
        }
      }
      return metrics[0].key;
    }

    function updateSubsetSummary(){
      if (!subsetSummaryBody) return;
      subsetSummaryBody.innerHTML = '';
      if (!subsetMetrics.length){
        var emptyRow = document.createElement('tr');
        var emptyCell = document.createElement('td');
        emptyCell.colSpan = 10;
        emptyCell.textContent = 'No subset statistics available for the current filters.';
        emptyRow.appendChild(emptyCell);
        subsetSummaryBody.appendChild(emptyRow);
        return;
      }
      var totalMetric = subsetMetrics.find(function(item){ return item.key === 'all'; });
      if (!totalMetric || !totalMetric.count){
        var noneRow = document.createElement('tr');
        var noneCell = document.createElement('td');
        noneCell.colSpan = 10;
        noneCell.textContent = 'No rows match the current filters.';
        noneRow.appendChild(noneCell);
        subsetSummaryBody.appendChild(noneRow);
        return;
      }
      SUBSET_TABLE_ORDER.forEach(function(key){
        var metric = subsetMetrics.find(function(item){ return item.key === key; });
        if (!metric) return;
        var tr = document.createElement('tr');
        tr.dataset.key = metric.key;
        if (metric.key === selectedSubsetKey){
          tr.classList.add('is-active');
        }
        tr.innerHTML = [
          '<td>' + metric.label + '</td>',
          '<td>' + metric.count + '</td>',
          '<td>' + formatPercent(metric.share) + '</td>',
          '<td>' + formatPercent(metric.accuracy) + '</td>',
          '<td>' + formatPercent(metric.multimodal_share) + '</td>',
          '<td>' + formatNumber(metric.points_summary && metric.points_summary.mean, 2) + '</td>',
          '<td>' + formatNumber(metric.points_earned_summary && metric.points_earned_summary.mean, 2) + '</td>',
          '<td>' + formatNumber(metric.latency_summary && metric.latency_summary.median, 1) + '</td>',
          '<td>' + formatNumber(metric.tokens_summary && metric.tokens_summary.mean, 0) + '</td>',
          '<td>' + formatCurrency(metric.cost_summary && metric.cost_summary.mean) + '</td>'
        ].join('');
        tr.addEventListener('click', function(){ selectSubset(metric.key); });
        subsetSummaryBody.appendChild(tr);
      });
    }

    function updateSubsetToggle(){
      if (!subsetToggle) return;
      subsetToggle.innerHTML = '';
      if (!subsetMetrics.length) return;
      SUBSET_TOGGLE_ORDER.forEach(function(key){
        var metric = subsetMetrics.find(function(item){ return item.key === key; });
        if (!metric) return;
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = metric.label + ' (' + metric.count + ')';
        if (metric.key === selectedSubsetKey){
          btn.classList.add('is-active');
        }
        if (!metric.count){
          btn.disabled = true;
        }
        btn.addEventListener('click', function(){ selectSubset(metric.key); });
        subsetToggle.appendChild(btn);
      });
    }

    function updateSubsetCharts(){
      if (!subsetUIEnabled) return;
      var metric = null;
      if (selectedSubsetKey){
        metric = subsetMetrics.find(function(item){ return item.key === selectedSubsetKey; }) || null;
      }
      if (!metric){
        metric = subsetMetrics.find(function(item){ return item.key === 'all'; }) || null;
      }
      if (!metric){
        if (subsetSummaryNote){
          subsetSummaryNote.textContent = 'No subset statistics available.';
        }
        if (gradeChart){
          charts.updateDistribution(gradeChart, 'subset-grade', []);
        }
        if (pointsChart){
          charts.updateHistogram(pointsChart, {});
        }
        if (pointsEarnedChart){
          charts.updateHistogram(pointsEarnedChart, {});
        }
        renderDistributionList(subsetLanguageList, []);
        renderDistributionList(subsetReasoningList, []);
        renderSubsetStats(null);
        return;
      }
      if (subsetSummaryNote){
        if (!metric.count){
          subsetSummaryNote.textContent = metric.label + ': no rows for the current filters.';
        } else {
          subsetSummaryNote.textContent = metric.label + ' · ' + metric.count + ' rows (' + formatPercent(metric.share) + ' of filtered)';
        }
      }
      if (gradeChart){
        charts.updateDistribution(gradeChart, 'subset-grade', metric.grade_distribution || []);
      }
      if (pointsChart){
        charts.updateHistogram(pointsChart, metric.points_hist || {});
      }
      if (pointsEarnedChart){
        charts.updateHistogram(pointsEarnedChart, metric.points_earned_hist || {});
      }
      renderDistributionList(subsetLanguageList, metric.language_distribution || []);
      renderDistributionList(subsetReasoningList, metric.reasoning_mode_distribution || []);
      renderSubsetStats(metric);
    }

    function refreshSubsetUI(){
      if (!subsetUIEnabled) return;
      updateSubsetSummary();
      updateSubsetToggle();
      updateSubsetCharts();
    }

    function selectSubset(key){
      if (!subsetUIEnabled) return;
      if (!key || key === selectedSubsetKey){
        return;
      }
      selectedSubsetKey = key;
      refreshSubsetUI();
    }

    function setSubsetMetrics(metrics){
      if (!subsetUIEnabled) return;
      subsetMetrics = Array.isArray(metrics) ? metrics.slice() : [];
      if (!subsetMetrics.length){
        selectedSubsetKey = null;
        refreshSubsetUI();
        return;
      }
      if (!selectedSubsetKey || !subsetMetrics.some(function(item){ return item.key === selectedSubsetKey && item.count > 0; })){
        selectedSubsetKey = findDefaultSubsetKey(subsetMetrics);
      }
      refreshSubsetUI();
    }

    function attachChipHandlers(){
      chipContainers.forEach(function(container){
        if (container.tagName === 'SELECT'){
          container.addEventListener('change', function(){
            var facet = container.dataset.facet;
            var field = facetToField(facet);
            state.filters[field] = Array.from(container.selectedOptions).map(function(option){ return option.value; });
            state.filters.page = 1;
            refresh();
          });
          return;
        }
        container.addEventListener('click', function(event){
          var target = event.target;
          if (target.tagName !== 'BUTTON') return;
          var facet = container.dataset.facet;
          var field = facetToField(facet);
          var list = state.filters[field];
          var value = target.dataset.value;
          var index = list.indexOf(value);
          if (index === -1){
            list.push(value);
            target.classList.add('active');
          } else {
            list.splice(index, 1);
            target.classList.remove('active');
          }
          state.filters.page = 1;
          refresh();
        });
      });
    }

    function buildQuery(){
      var params = {};
      Object.keys(state.filters).forEach(function(key){
        var value = state.filters[key];
        if (Array.isArray(value)){
          if (value.length){
            params[key] = value;
          }
        } else if (value !== '' && value !== null && value !== undefined){
          params[key] = value;
        }
      });
      params.page = state.filters.page;
      params.page_size = state.filters.page_size;
      params.sort_by = state.filters.sort_by;
      params.sort_dir = state.filters.sort_dir;
      return params;
    }

    function refresh(){
      renderActiveFilters();
      loadResults();
      loadAggregates();
    }

    function loadFacets(){
      fetchJSON('/api/runs/' + runId + '/facets')
        .then(function(facets){
          chipContainers.forEach(function(container){
            var facet = container.dataset.facet;
            var field = facetToField(facet);
            var values = facets[facet] || [];
            if (container.tagName === 'SELECT'){
              container.innerHTML = '';
              values.forEach(function(value){
                var option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                container.appendChild(option);
              });
            } else {
              container.innerHTML = '';
              values.forEach(function(value){
                var btn = document.createElement('button');
                btn.type = 'button';
                btn.textContent = value;
                btn.dataset.value = value;
                container.appendChild(btn);
              });
            }
          });
          attachChipHandlers();
          syncControls();
        })
        .catch(function(err){ console.error('Failed to load facets', err); });
    }

    function loadResults(){
      fetchJSON('/api/runs/' + runId + '/results', buildQuery())
        .then(function(page){
          renderTable(page.items || []);
          renderPagination(page.page, page.page_size, page.total);
        })
        .catch(function(err){
          console.error('Failed to load results', err);
        });
    }

    function renderTable(items){
      tableBody.innerHTML = '';
      if (!items.length){
        var empty = document.createElement('tr');
        var cell = document.createElement('td');
        cell.colSpan = 14;
        cell.textContent = 'No rows match the current filters.';
        empty.appendChild(cell);
        tableBody.appendChild(empty);
        return;
      }
      items.forEach(function(row){
        var tr = document.createElement('tr');
        tr.innerHTML = [
          row.id,
          row.group || '–',
          row.year || '–',
          row.problem_number || '–',
          row.points ?? '–',
          row.answer || '–',
          row.predicted || '–',
          row.is_correct === true ? '✔' : row.is_correct === false ? '✖' : '–',
          row.points_earned ?? '–',
          row.latency_ms ? row.latency_ms.toFixed(1) : '–',
          row.total_tokens ?? '–',
          row.reasoning_tokens ?? '–',
          row.cost_usd ? row.cost_usd.toFixed(4) : '–',
          row.warnings && row.warnings.length ? '⚠ ' + row.warnings.length : '–'
        ].map(function(value){ return '<td>' + value + '</td>'; }).join('');
        tr.addEventListener('click', function(){ openRow(row.id); });
        tableBody.appendChild(tr);
      });
    }

    function renderPagination(page, pageSizeValue, total){
      pagination.innerHTML = '';
      var pageCount = Math.max(1, Math.ceil(total / pageSizeValue));

      var createButton = function(label, newPage, options){
        var opts = options || {};
        var btn = document.createElement('button');
        btn.textContent = label;
        if (opts.isEllipsis){
          btn.disabled = true;
          btn.classList.add('ellipsis');
        } else if (opts.disabled){
          btn.disabled = true;
        } else {
          btn.addEventListener('click', function(){
            state.filters.page = newPage;
            refresh();
          });
        }
        if (opts.active){
          btn.classList.add('active');
        }
        pagination.appendChild(btn);
      };

      createButton('Prev', Math.max(1, page - 1), { disabled: page <= 1 });

      var windowRadius = 2;
      var start = Math.max(2, page - windowRadius);
      var end = Math.min(pageCount - 1, page + windowRadius);

      var addPage = function(pageNumber){
        createButton(String(pageNumber), pageNumber, { active: pageNumber === page });
      };

      addPage(1);

      if (start > 2){
        createButton('…', null, { isEllipsis: true });
      }

      for (var i = start; i <= end; i++){
        addPage(i);
      }

      if (end < pageCount - 1){
        createButton('…', null, { isEllipsis: true });
      }

      if (pageCount > 1){
        addPage(pageCount);
      }

      createButton('Next', Math.min(pageCount, page + 1), { disabled: page >= pageCount });
    }

    function loadAggregates(){
      fetchJSON('/api/runs/' + runId + '/aggregates', buildQuery())
        .then(function(data){
          lastAggregates = data;
          if (subsetUIEnabled){
            setSubsetMetrics(data.subset_metrics || []);
          }
          charts.updateBreakdown($('#chart-group'), 'chart-group', data.breakdown_by_group || {});
          charts.updateBreakdown($('#chart-year'), 'chart-year', data.breakdown_by_year || {});
          charts.updateConfusion($('#chart-confusion'), data.confusion_matrix || {});
          charts.updateHistogram($('#chart-latency'), data.latency_hist || {});
          charts.updateHistogram($('#chart-token'), data.tokens_hist || {});
          charts.updatePredicted($('#chart-predicted'), data.predicted_counts || {});
          renderWarningList(data.warning_toplist || []);
        })
        .catch(function(err){ console.error('Failed to load aggregates', err); });
    }

    function updateHumanCards(entry){
      if (!humanCards) return;
      if (!entry){
        humanCards.hidden = true;
        return;
      }
      humanCards.hidden = false;
      if (humanPercentileValue){
        humanPercentileValue.textContent = formatPercent(entry.human_percentile);
      }
      if (humanPercentileNote){
        humanPercentileNote.textContent = entry.grade_label || entry.grade_id || '';
      }
      if (humanZScoreValue){
        if (entry.z_score !== null && entry.z_score !== undefined){
          humanZScoreValue.textContent = entry.z_score.toFixed(2);
        } else {
          humanZScoreValue.textContent = '–';
        }
      }
      if (humanZScoreNote){
        humanZScoreNote.textContent = entry.human_std ? 'σ ≈ ' + entry.human_std.toFixed(2) : '';
      }
      if (humanScoreValue){
        if (entry.llm_total !== null && entry.llm_max !== null){
          humanScoreValue.textContent = formatNumber(entry.llm_total, 1) + ' / ' + formatNumber(entry.llm_max, 1);
        } else {
          humanScoreValue.textContent = '–';
        }
      }
      if (humanScoreNote){
        humanScoreNote.textContent = entry.grade_label || entry.grade_id || '';
      }
    }

    function loadHumanComparison(){
      fetchJSON('/api/humans/compare/run/' + runId)
        .then(function(payload){
          state.humanComparison = payload;
          if (!payload || !payload.entries || !payload.entries.length){
            updateHumanCards(null);
            return;
          }
          var sorted = payload.entries.slice().sort(function(a, b){
            var pa = a.human_percentile || 0;
            var pb = b.human_percentile || 0;
            return pb - pa;
          });
          updateHumanCards(sorted[0]);
        })
        .catch(function(){ updateHumanCards(null); });
    }

    function rerenderCharts(){
      if (!lastAggregates) return;
      charts.updateBreakdown($('#chart-group'), 'chart-group', lastAggregates.breakdown_by_group || {});
      charts.updateBreakdown($('#chart-year'), 'chart-year', lastAggregates.breakdown_by_year || {});
      charts.updateConfusion($('#chart-confusion'), lastAggregates.confusion_matrix || {});
      charts.updateHistogram($('#chart-latency'), lastAggregates.latency_hist || {});
      charts.updateHistogram($('#chart-token'), lastAggregates.tokens_hist || {});
      charts.updatePredicted($('#chart-predicted'), lastAggregates.predicted_counts || {});
      if (subsetUIEnabled){
        updateSubsetCharts();
      }
    }

    function renderWarningList(entries){
      if (!warningList) return;
      warningList.innerHTML = '';
      if (!entries.length){
        var li = document.createElement('li');
        li.textContent = 'No warnings recorded.';
        warningList.appendChild(li);
        return;
      }
      entries.forEach(function(entry){
        var li = document.createElement('li');
        li.textContent = entry.warning_type + ' · ' + entry.count;
        li.addEventListener('click', function(){
          if (state.filters.warning_types.indexOf(entry.warning_type) === -1){
            state.filters.warning_types.push(entry.warning_type);
            state.filters.warnings_present = 'true';
            syncControls();
            refresh();
          }
        });
        warningList.appendChild(li);
      });
    }

    function loadFailures(){
      fetchJSON('/api/runs/' + runId + '/failures')
        .then(function(entries){
          failuresList.innerHTML = '';
          if (!entries.length){
            var li = document.createElement('li');
            li.textContent = 'No failures recorded.';
            failuresList.appendChild(li);
            return;
          }
          entries.forEach(function(entry){
            var li = document.createElement('li');
            var summary = (entry.timestamp || 'n/a') + ' · ' + (entry.status_code || 'code?') + ' · ' + (entry.message || '');
            var summarySpan = document.createElement('span');
            summarySpan.textContent = summary;
            li.appendChild(summarySpan);
            if (entry.id){
              li.appendChild(document.createTextNode(' '));
              var link = document.createElement('button');
              link.type = 'button';
              link.className = 'link-button';
              link.textContent = 'View ' + entry.id;
              link.addEventListener('click', function(){
                openRow(entry.id);
              });
              li.appendChild(link);
            }
            failuresList.appendChild(li);
          });
        })
        .catch(function(err){ console.error('Failed to load failures', err); });
    }

    function openRow(rowId){
      fetchJSON('/api/runs/' + runId + '/row/' + encodeURIComponent(rowId))
        .then(function(payload){
          renderRowDrawer(payload);
        })
        .catch(function(err){
          console.error('Failed to load row detail', err);
        });
    }

    function renderRowDrawer(payload){
      rowMeta.innerHTML = '';
      var row = payload.row;
      var dataset = payload.dataset;
      var info = document.createElement('div');
      info.innerHTML = [
        '<strong>ID:</strong> ' + row.id,
        '<strong>Group:</strong> ' + (row.group || '–'),
        '<strong>Year:</strong> ' + (row.year || '–'),
        '<strong>Answer:</strong> ' + (row.answer || '–'),
        '<strong>Predicted:</strong> ' + (row.predicted || '–'),
        '<strong>Latency:</strong> ' + (row.latency_ms ? row.latency_ms.toFixed(1) + ' ms' : '–'),
        '<strong>Total tokens:</strong> ' + (row.total_tokens ?? '–'),
        '<strong>Reasoning tokens:</strong> ' + (row.reasoning_tokens ?? '–'),
        '<strong>Cost:</strong> ' + (row.cost_usd ? row.cost_usd.toFixed(4) : '–'),
        row.warnings && row.warnings.length ? '<strong>Warnings:</strong> ' + row.warnings.join(', ') : ''
      ].filter(Boolean).map(function(line){ return '<p>' + line + '</p>'; }).join('');
      rowMeta.appendChild(info);
      if (rowRationale){
        if (row.rationale){
          rowRationale.classList.remove('hidden');
          rowRationale.innerHTML = '<h3>Rationale</h3>';
          var rationaleBlock = document.createElement('pre');
          rationaleBlock.className = 'row-rationale__text';
          rationaleBlock.textContent = row.rationale;
          rowRationale.appendChild(rationaleBlock);
        } else {
          rowRationale.classList.add('hidden');
          rowRationale.innerHTML = '';
        }
      }
      if (rowRaw && rawToggle && rawResponse){
        if (row.raw_text_response){
          rowRaw.classList.remove('hidden');
          rawToggle.classList.remove('hidden');
          rawResponse.textContent = row.raw_text_response;
          rawResponse.classList.add('hidden');
          rawToggle.textContent = 'Show raw response';
          rawToggle.onclick = function(){
            var isHidden = rawResponse.classList.toggle('hidden');
            rawToggle.textContent = isHidden ? 'Show raw response' : 'Hide raw response';
          };
        } else {
          rowRaw.classList.add('hidden');
          rawToggle.classList.add('hidden');
          rawResponse.textContent = '';
          rawResponse.classList.add('hidden');
          rawToggle.onclick = null;
        }
      }
      rowDataset.innerHTML = '';
      if (dataset){
        var problem = document.createElement('article');
        problem.innerHTML = '<h3>Problem statement</h3><p>' + (dataset.problem_statement || 'n/a') + '</p>';
        rowDataset.appendChild(problem);
        if (dataset.question_image){
          var img = document.createElement('img');
          img.src = dataset.question_image;
          img.alt = 'Question image';
          rowDataset.appendChild(img);
        }
        if (dataset.options && dataset.options.length){
          var list = document.createElement('ul');
          dataset.options.forEach(function(option){
            var item = document.createElement('li');
            var content = '<strong>' + option.label + '.</strong> ' + (option.text || '');
            if (option.image){
              content += '<br><img src="' + option.image + '" alt="Option ' + option.label + '" />';
            }
            item.innerHTML = content;
            list.appendChild(item);
          });
          rowDataset.appendChild(list);
        }
        if (dataset.associated_images && dataset.associated_images.length){
          dataset.associated_images.forEach(function(src){
            var img = document.createElement('img');
            img.src = src;
            img.alt = 'Associated image';
            rowDataset.appendChild(img);
          });
        }
      }
      rowDrawer.classList.remove('hidden');
      rowDrawer.classList.add('visible');
    }

    function closeDrawer(){
      rowDrawer.classList.remove('visible');
      rowDrawer.classList.add('hidden');
    }

    if (closeRowBtn){
      closeRowBtn.addEventListener('click', closeDrawer);
    }

    function attachSelectChange(select, key){
      if (!select) return;
      select.addEventListener('change', function(){
        state.filters[key] = select.value;
        state.filters.page = 1;
        refresh();
      });
    }

    function attachRangeInput(field){
      var input = filterForm.elements[field];
      if (!input) return;
      input.addEventListener('change', function(){
        state.filters[field] = input.value;
        state.filters.page = 1;
        refresh();
      });
    }

    filterForm.addEventListener('submit', function(event){
      event.preventDefault();
      state.filters.multimodal = filterForm.multimodal.value;
      state.filters.correctness = filterForm.correctness.value;
      state.filters.warnings_present = filterForm.warnings_present.value;
      ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','cost_min','cost_max'].forEach(function(field){
        state.filters[field] = filterForm.elements[field].value;
      });
      state.filters.page = 1;
      refresh();
    });

    attachSelectChange(filterForm.multimodal, 'multimodal');
    attachSelectChange(filterForm.correctness, 'correctness');
    attachSelectChange(filterForm.warnings_present, 'warnings_present');
    ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','cost_min','cost_max'].forEach(attachRangeInput);

    sortBy.addEventListener('change', function(){
      state.filters.sort_by = sortBy.value;
      refresh();
    });

    sortDir.addEventListener('change', function(){
      state.filters.sort_dir = sortDir.value;
      refresh();
    });

    pageSize.addEventListener('change', function(){
      state.filters.page_size = parseInt(pageSize.value, 10) || 25;
      state.filters.page = 1;
      refresh();
    });

    resetBtn.addEventListener('click', function(){
      Object.keys(state.filters).forEach(function(key){
        if (Array.isArray(state.filters[key])){
          state.filters[key] = [];
        } else if (['page','page_size'].indexOf(key) !== -1){
          // preserve pagination defaults
        } else if (key === 'sort_by'){
          state.filters[key] = 'id';
        } else if (key === 'sort_dir'){
          state.filters[key] = 'asc';
        } else if (key === 'page_size'){
          state.filters[key] = 25;
        } else {
          state.filters[key] = '';
        }
      });
      state.filters.page = 1;
      state.filters.page_size = 25;
      syncControls();
      refresh();
    });

    if (exportJson){
      exportJson.addEventListener('click', function(){
        triggerDownload('json');
      });
    }
    if (exportCsv){
      exportCsv.addEventListener('click', function(){
        triggerDownload('csv');
      });
    }

    function triggerDownload(format){
      var params = buildQuery();
      params.download = format;
      var query = new URLSearchParams();
      Object.keys(params).forEach(function(key){
        var value = params[key];
        if (Array.isArray(value)){
          value.forEach(function(v){ query.append(key, v); });
        } else {
          query.append(key, value);
        }
      });
      window.location.href = '/api/runs/' + runId + '/results?' + query.toString();
    }

    if (savePresetBtn){
      savePresetBtn.addEventListener('click', function(){
        var name = (presetName.value || '').trim();
        if (!name){
          alert('Enter a preset name');
          return;
        }
        var presets = loadPresets();
        presets[name] = serializeFilters();
        savePresets(presets);
        updatePresetSelect();
      });
    }

    if (loadPresetBtn){
      loadPresetBtn.addEventListener('click', function(){
        var name = presetSelect.value;
        if (!name) return;
        var presets = loadPresets();
        if (presets[name]){
          restoreFilters(presets[name]);
        }
      });
    }

    updatePresetSelect();
    loadFacets();
    loadFailures();
    syncControls();
    refresh();
    loadHumanComparison();

    document.addEventListener('dashboard:themechange', rerenderCharts);
  }

  function initCompare(){
    var compareSection = document.querySelector('.compare');
    if (!compareSection) return;
    var runAInput = $('#compare-run-a');
    var runBInput = $('#compare-run-b');
    var form = $('#compare-form');
    var viewSelect = $('#compare-view');
    var limitInput = $('#compare-limit');
    var refreshBtn = $('#compare-refresh');
    var compareTableBody = $('#compare-table tbody');
    var charts = new window.DashboardCharts();
    var lastCompare = null;
    var confusionToggle = $('#compare-confusion-toggle');
    var confusionSection = $('#compare-confusion-section');
    var confusionLabelA = $('#compare-confusion-label-a');
    var confusionLabelB = $('#compare-confusion-label-b');
    var confusionCanvasA = $('#compare-confusion-a');
    var confusionCanvasB = $('#compare-confusion-b');

    function redirect(){
      var params = new URLSearchParams();
      params.append('run_a', runAInput.value);
      params.append('run_b', runBInput.value);
      window.location.href = '/compare?' + params.toString();
    }

    if (form){
      form.addEventListener('submit', function(event){
        event.preventDefault();
        if (!runAInput.value || !runBInput.value) return;
        redirect();
      });
    }

    function fetchCompare(){
      if (!runAInput.value || !runBInput.value) return;
      var params = {
        run_a: runAInput.value,
        run_b: runBInput.value,
        view: viewSelect ? viewSelect.value : 'all'
      };
      var limitVal = limitInput && limitInput.value ? parseInt(limitInput.value, 10) : null;
      if (limitVal){
        params.limit = limitVal;
      }
      if (confusionToggle && confusionToggle.checked){
        params.include_confusion = 'true';
      } else if (confusionSection){
        confusionSection.classList.add('hidden');
      }
      fetchJSON('/api/compare', params)
        .then(function(payload){
          renderCompare(payload);
        })
        .catch(function(err){ console.error('Failed to fetch compare', err); });
    }

    function renderCompare(payload){
      lastCompare = payload;
      var metricGrid = $('#compare-metrics');
      if (metricGrid && payload.metrics){
        metricGrid.innerHTML = '';
        payload.metrics.forEach(function(metric){
          var deltaDisplay = '–';
          if (metric.delta !== null && metric.delta !== undefined){
            var deltaNumber = Number(metric.delta);
            deltaDisplay = isNaN(deltaNumber) ? '–' : deltaNumber.toFixed(4);
          }
          var card = document.createElement('div');
          card.className = 'summary-card';
          card.innerHTML = '<span class="summary-label">' + metric.metric.replace(/_/g, ' ') + '</span>' +
            '<span class="summary-value">' + deltaDisplay + '</span>' +
            '<span class="summary-note">A: ' + (metric.run_a ?? '–') + ' · B: ' + (metric.run_b ?? '–') + '</span>';
          metricGrid.appendChild(card);
        });
      }
      if (payload.breakdown_deltas){
        charts.updateCompareSeries($('#compare-group'), payload.breakdown_deltas.group || []);
        charts.updateCompareSeries($('#compare-year'), payload.breakdown_deltas.year || []);
      }
      if (payload.row_deltas && compareTableBody){
        compareTableBody.innerHTML = '';
        payload.row_deltas.forEach(function(row){
          var tr = document.createElement('tr');
          tr.innerHTML = [
            row.id,
            row.run_a_correct,
            row.run_b_correct,
            row.run_a_predicted || '–',
            row.run_b_predicted || '–',
            row.run_a_points ?? '–',
            row.run_b_points ?? '–',
            row.delta_points ?? '–',
            row.delta_latency_ms ?? '–',
            row.delta_total_tokens ?? '–'
          ].map(function(value){ return '<td>' + value + '</td>'; }).join('');
          compareTableBody.appendChild(tr);
        });
      }
      if (confusionSection){
        var hasConfusion = payload.confusion_matrices && payload.confusion_matrices.run_a && payload.confusion_matrices.run_b;
        if (confusionToggle && confusionToggle.checked && hasConfusion){
          confusionSection.classList.remove('hidden');
          if (confusionLabelA){
            confusionLabelA.textContent = payload.run_a.model_label || payload.run_a.model_id || payload.run_a.run_id;
          }
          if (confusionLabelB){
            confusionLabelB.textContent = payload.run_b.model_label || payload.run_b.model_id || payload.run_b.run_id;
          }
          if (confusionCanvasA){
            charts.updateConfusion(confusionCanvasA, payload.confusion_matrices.run_a);
          }
          if (confusionCanvasB){
            charts.updateConfusion(confusionCanvasB, payload.confusion_matrices.run_b);
          }
        } else {
          confusionSection.classList.add('hidden');
        }
      }
    }

    function rerenderCompareCharts(){
      if (!lastCompare) return;
      if (lastCompare.breakdown_deltas){
        charts.updateCompareSeries($('#compare-group'), lastCompare.breakdown_deltas.group || []);
        charts.updateCompareSeries($('#compare-year'), lastCompare.breakdown_deltas.year || []);
      }
      if (!confusionSection) return;
      var hasConfusion = lastCompare.confusion_matrices && lastCompare.confusion_matrices.run_a && lastCompare.confusion_matrices.run_b;
      if (confusionToggle && confusionToggle.checked && hasConfusion){
        confusionSection.classList.remove('hidden');
        if (confusionLabelA){
          confusionLabelA.textContent = lastCompare.run_a.model_label || lastCompare.run_a.model_id || lastCompare.run_a.run_id;
        }
        if (confusionLabelB){
          confusionLabelB.textContent = lastCompare.run_b.model_label || lastCompare.run_b.model_id || lastCompare.run_b.run_id;
        }
        charts.updateConfusion(confusionCanvasA, lastCompare.confusion_matrices.run_a);
        charts.updateConfusion(confusionCanvasB, lastCompare.confusion_matrices.run_b);
      }
    }

    if (refreshBtn){
      refreshBtn.addEventListener('click', fetchCompare);
    }
    if (viewSelect){
      viewSelect.addEventListener('change', fetchCompare);
    }
    if (limitInput){
      limitInput.addEventListener('change', fetchCompare);
    }
    if (confusionToggle){
      confusionToggle.addEventListener('change', function(){
        if (confusionToggle.checked){
          fetchCompare();
        } else if (confusionSection){
          confusionSection.classList.add('hidden');
        }
      });
    }

    // Auto fetch if selections present
    if (runAInput.value && runBInput.value){
      fetchCompare();
    }

    document.addEventListener('dashboard:themechange', rerenderCompareCharts);
  }

  function initAnalysis(){
    var analysisSection = document.querySelector('.analysis');
    if (!analysisSection) return;

    var form = $('#analysis-form');
    var resultsContainer = $('#analysis-results-container');
    var chartsContainer = $('.analysis-charts');
    var tagsContainer = $('.analysis-tags');
    var scatterPlotEl = $('#scatter-human-vs-llm');
    var barChartEl = $('#bar-normalized-performance');
    var tagTableBody = $('#tag-analysis-table tbody');
    var charts = new window.DashboardCharts();

    form.addEventListener('submit', function(event){
      event.preventDefault();
      var formData = new FormData(form);
      var runIds = formData.getAll('run_ids');
      if (!runIds.length) {
        alert('Please select at least one run.');
        return;
      }

      resultsContainer.style.display = 'block';
      $('#analysis-results').innerHTML = '<p>Loading analysis...</p>';
      chartsContainer.style.display = 'none';
      tagsContainer.style.display = 'none';

      // Construct form data for POST request
      var postData = new URLSearchParams();
      runIds.forEach(function(id) { postData.append('run_ids', id); });

      fetch('/api/analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: postData
      })
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Request failed: ' + response.status);
        }
        return response.json();
      })
      .then(function(data){
          $('#analysis-results').innerHTML = ''; // Clear loading message
          chartsContainer.style.display = 'block';
          tagsContainer.style.display = 'block';
          
          var scatterData = data.scatter.map(function(item) {
            return [item.human_p_correct, item.avg_llm_score, item.question_id, item.llm_disagreement];
          }).filter(function(item) {
            return item[0] !== null && item[0] !== undefined;
          });

          charts.updateScatter(scatterPlotEl, scatterData);
          charts.updateBar(barChartEl, data.bars);
          renderTagTable(data.tags);
        })
        .catch(function(err){
          $('#analysis-results').innerHTML = '<p class="error">Failed to load analysis: ' + err.message + '</p>';
          console.error('Failed to load analysis', err);
        });
    });

    function renderTagTable(tags) {
      if (!tagTableBody) return;
      tagTableBody.innerHTML = '';
      if (!tags || !tags.length) {
        var row = tagTableBody.insertRow();
        var cell = row.insertCell();
        cell.colSpan = 4;
        cell.textContent = 'No tags found for this selection.';
        return;
      }

      tags.forEach(function(tag) {
        var row = tagTableBody.insertRow();
        row.insertCell().textContent = tag.tag;
        row.insertCell().textContent = tag.count;
        row.insertCell().textContent = (tag.avg_human_score * 100).toFixed(1) + '%';
        row.insertCell().textContent = (tag.avg_llm_score * 100).toFixed(1) + '%';
      });
    }
  }

  function initHumans(){
    var root = document.getElementById('human-baseline-root');
    if (!root) return;
    if (root.dataset.hasData !== 'true') return;

    var bootstrapEl = document.getElementById('human-baseline-bootstrap');
    var bootstrap = { runs: [], years: [] };
    if (bootstrapEl && bootstrapEl.textContent){
      try {
        bootstrap = JSON.parse(bootstrapEl.textContent);
      } catch (err){
        console.error('Failed to parse human baseline bootstrap', err);
      }
    }

    var charts = new window.DashboardCharts();
    var state = {
      runs: bootstrap.runs || [],
      years: bootstrap.years || [],
      yearMap: {},
      selectedView: 'run',
      selectedRun: null,
      selectedRuns: [],
      selectedYear: null,
      selectedGrade: null,
      selectedCohortType: 'micro',
      runComparisons: {},
      cohortCache: {},
      yearSummaryCache: {},
      cdfCache: {},
      percentileCache: {},
    };

    state.years.forEach(function(entry){
      if (entry && entry.year !== undefined){
        state.yearMap[entry.year] = entry;
      }
    });

    var viewSelect = $('#human-view-select');
    var runSelect = $('#human-run-select');
    var runSelectWrapper = $('#human-run-select-wrapper');
    var runMulti = $('#human-run-multi');
    var runMultiWrapper = $('#human-run-multi-wrapper');
    var yearSelect = $('#human-year-select');
    var gradeSelect = $('#human-grade-select');
    var cohortTypeSelect = $('#human-cohort-type');
    var cohortTypeWrapper = $('#human-cohort-type-wrapper');
    var noteEl = $('#human-baseline-note');

    var runView = $('#human-run-view');
    var cohortView = $('#human-cohort-view');
    var chartsSection = $('#human-charts');

    var runTableBody = $('#human-run-table-body');
    var cohortTableBody = $('#human-cohort-table-body');

    var runPercentile = $('#human-run-percentile');
    var runPercentileNote = $('#human-run-percentile-note');
    var runZScore = $('#human-run-zscore');
    var runZScoreNote = $('#human-run-zscore-note');
    var runScore = $('#human-run-score');
    var runScoreNote = $('#human-run-score-note');

    var cdfCanvas = $('#human-cdf-chart');
    var heatmapCanvas = $('#human-heatmap-chart');
    var percentileCanvas = $('#human-percentile-chart');

    function labelForRun(run){
      if (!run) return '';
      return (run.model_label || run.model_id || run.run_id);
    }

    function setSelectedOption(select, value){
      if (!select) return;
      var has = false;
      Array.from(select.options).forEach(function(option){
        if (option.value === value){
          option.selected = true;
          has = true;
        } else if (!select.multiple){
          option.selected = false;
        }
      });
      if (!has && select.options.length){
        select.options[0].selected = true;
      }
    }

    function populateRunSelectors(){
      if (runSelect){
        runSelect.innerHTML = '';
      }
      if (runMulti){
        runMulti.innerHTML = '';
      }
      state.runs.forEach(function(run){
        var label = labelForRun(run) + ' · ' + run.run_id;
        if (runSelect){
          var opt = document.createElement('option');
          opt.value = run.run_id;
          opt.textContent = label;
          runSelect.appendChild(opt);
        }
        if (runMulti){
          var optMulti = document.createElement('option');
          optMulti.value = run.run_id;
          optMulti.textContent = label;
          runMulti.appendChild(optMulti);
        }
      });
      if (!state.selectedRun && state.runs.length){
        state.selectedRun = state.runs[0].run_id;
      }
      if (runSelect && state.selectedRun){
        setSelectedOption(runSelect, state.selectedRun);
      }
    }

    function populateYearSelect(){
      if (!yearSelect) return;
      var years = state.years.slice().sort(function(a, b){ return a.year - b.year; });
      yearSelect.innerHTML = '';
      years.forEach(function(entry){
        var option = document.createElement('option');
        option.value = String(entry.year);
        option.textContent = String(entry.year);
        yearSelect.appendChild(option);
      });
      if (!state.selectedYear && years.length){
        state.selectedYear = years[0].year;
      }
      if (state.selectedYear && yearSelect.options.length){
        setSelectedOption(yearSelect, String(state.selectedYear));
      }
    }

    function populateGradeSelect(year){
      if (!gradeSelect) return;
      gradeSelect.innerHTML = '';
      var entry = state.yearMap[year];
      if (!entry || !entry.grades){
        state.selectedGrade = null;
        return;
      }
      entry.grades.forEach(function(grade){
        var option = document.createElement('option');
        option.value = grade.id;
        option.textContent = grade.label || grade.id;
        option.dataset.members = JSON.stringify(grade.members || []);
        gradeSelect.appendChild(option);
      });
      if (!state.selectedGrade && entry.grades.length){
        state.selectedGrade = entry.grades[0].id;
      }
      if (state.selectedGrade){
        setSelectedOption(gradeSelect, state.selectedGrade);
      }
    }

    function getSelectedRuns(){
      if (!runMulti) return [];
      return Array.from(runMulti.selectedOptions).map(function(opt){ return opt.value; });
    }

    function ensureCohortDefaults(){
      if (!state.selectedRuns.length && state.runs.length){
        state.selectedRuns = state.runs.slice(0, Math.min(3, state.runs.length)).map(function(run){ return run.run_id; });
        if (runMulti){
          Array.from(runMulti.options).forEach(function(opt){
            opt.selected = state.selectedRuns.indexOf(opt.value) !== -1;
          });
        }
      }
    }

    function fetchRunComparison(runId){
      if (state.runComparisons[runId]){
        return Promise.resolve(state.runComparisons[runId]);
      }
      return fetchJSON('/api/humans/compare/run/' + encodeURIComponent(runId))
        .then(function(payload){
          state.runComparisons[runId] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load human comparison for run', runId, err);
          return null;
        });
    }

    function fetchYearSummary(year){
      if (state.yearSummaryCache[year]){
        return Promise.resolve(state.yearSummaryCache[year]);
      }
      return fetchJSON('/api/humans/' + year + '/summary')
        .then(function(payload){
          state.yearSummaryCache[year] = payload;
          return payload;
        });
    }

    function fetchCdf(year, gradeId){
      var key = year + ':' + gradeId;
      if (state.cdfCache[key]){
        return Promise.resolve(state.cdfCache[key]);
      }
      return fetchJSON('/api/humans/' + year + '/cdf', { grade: gradeId })
        .then(function(payload){
          state.cdfCache[key] = payload;
          return payload;
        });
    }

    function fetchPercentile(year, gradeId, score){
      var key = year + ':' + gradeId + ':' + score;
      if (state.percentileCache[key]){
        return Promise.resolve(state.percentileCache[key]);
      }
      return fetchJSON('/api/humans/percentile', { year: year, grade: gradeId, score: score })
        .then(function(payload){
          state.percentileCache[key] = payload;
          return payload;
        });
    }

    function fetchCohort(runIds){
      var key = runIds.slice().sort().join('|');
      if (!key){
        return Promise.resolve(null);
      }
      if (state.cohortCache[key]){
        return Promise.resolve(state.cohortCache[key]);
      }
      return fetch('/api/humans/compare/aggregate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_ids: runIds })
      })
        .then(function(response){
          if (!response.ok){
            throw new Error('Request failed: ' + response.status);
          }
          return response.json();
        })
        .then(function(payload){
          state.cohortCache[key] = payload;
          return payload;
        })
        .catch(function(err){
          console.error('Failed to load cohort comparison', err);
          return null;
        });
    }

    function formatPercent(value){
      if (value === null || value === undefined || isNaN(value)) return '–';
      return (Number(value) * 100).toFixed(1).replace(/\.0$/, '') + '%';
    }

    function formatScore(value){
      if (value === null || value === undefined || isNaN(value)) return '–';
      return Number(value).toFixed(1).replace(/\.0$/, '');
    }

    function updateRunSummary(entry){
      if (!entry){
        runPercentile.textContent = '–';
        runPercentileNote.textContent = '';
        runZScore.textContent = '–';
        runZScoreNote.textContent = '';
        runScore.textContent = '–';
        runScoreNote.textContent = '';
        return;
      }
      if (entry.human_percentile !== null && entry.human_percentile !== undefined){
        runPercentile.textContent = formatPercent(entry.human_percentile);
        runPercentileNote.textContent = entry.grade_label;
      } else {
        runPercentile.textContent = '–';
        runPercentileNote.textContent = 'Human percentile unavailable';
      }
      if (entry.z_score !== null && entry.z_score !== undefined){
        runZScore.textContent = entry.z_score.toFixed(2);
        runZScoreNote.textContent = entry.grade_label;
      } else {
        runZScore.textContent = '–';
        runZScoreNote.textContent = 'Human variance unavailable';
      }
      if (entry.llm_total !== null && entry.llm_max !== null){
        runScore.textContent = formatScore(entry.llm_total) + ' / ' + formatScore(entry.llm_max);
        runScoreNote.textContent = entry.grade_label;
      } else {
        runScore.textContent = '–';
        runScoreNote.textContent = '';
      }
    }

    function renderRunTable(runData){
      if (!runTableBody) return;
      runTableBody.innerHTML = '';
      if (!runData || !runData.entries){
        return;
      }
      runData.entries.forEach(function(entry){
        var tr = document.createElement('tr');
        tr.innerHTML = [
          '<td>' + entry.year + '</td>',
          '<td>' + (entry.grade_label || entry.grade_id) + '</td>',
          '<td>' + formatScore(entry.llm_total) + '</td>',
          '<td>' + formatScore(entry.llm_max) + '</td>',
          '<td>' + formatPercent(entry.llm_score_pct) + '</td>',
          '<td>' + formatPercent(entry.human_percentile) + '</td>',
          '<td>' + (entry.z_score !== null && entry.z_score !== undefined ? entry.z_score.toFixed(2) : '–') + '</td>'
        ].join('');
        runTableBody.appendChild(tr);
      });
    }

    function buildHeatmapData(entries, label){
      if (!entries || !entries.length) return null;
      var entry = entries[0];
      var labels = entry.bin_comparison.map(function(item){ return item.bin_id; });
      var values = entry.bin_comparison.map(function(item, idx){
        return [idx, 0, Number(item.delta_smoothed || item.delta || 0) * 100];
      });
      return {
        xLabels: labels,
        yLabels: [label],
        values: values,
      };
    }

    function updateRunCharts(runData, entry){
      if (!entry){
        charts.updateCDF(cdfCanvas, [], [], 0);
        charts.updateHeatmap(heatmapCanvas, { xLabels: [], yLabels: [], values: [] });
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = true;
        return;
      }
      var year = entry.year;
      var gradeId = entry.grade_id;
      Promise.all([
        fetchYearSummary(year),
        fetchCdf(year, gradeId),
        (entry.human_percentile === null || entry.human_percentile === undefined)
          ? fetchPercentile(year, gradeId, entry.llm_total)
          : Promise.resolve(null)
      ]).then(function(results){
        var yearSummary = results[0];
        var cdf = results[1];
        var percentileOverride = results[2];
        if (percentileOverride && percentileOverride.percentile !== undefined){
          entry.human_percentile = percentileOverride.percentile;
        }

        var markers = [];
        if (entry.human_percentile !== null && entry.human_percentile !== undefined){
          markers.push({
            label: labelForRun(state.runs.find(function(run){ return run.run_id === runData.run_id; })),
            score: entry.llm_total,
            percentile: entry.human_percentile,
          });
        }

        charts.updateCDF(
          cdfCanvas,
          (cdf && cdf.points) ? cdf.points.map(function(point){
            return { score: point.score, percentile: point.percentile };
          }) : [],
          markers,
          yearSummary && yearSummary.grades ? entry.max_points : entry.llm_max
        );

        var heatmapData = buildHeatmapData([entry], entry.grade_label || entry.grade_id);
        charts.updateHeatmap(heatmapCanvas, heatmapData);
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = false;
      }).catch(function(err){
        console.error('Failed to update run charts', err);
        chartsSection.hidden = false;
      });
    }

    function renderRunView(){
      if (!state.selectedRun){
        runView.hidden = true;
        return;
      }
      runView.hidden = false;
      var comparison = state.runComparisons[state.selectedRun];
      if (!comparison || !comparison.entries || !comparison.entries.length){
        noteEl.textContent = 'No baseline data available for this run.';
        renderRunTable(null);
        updateRunSummary(null);
        updateRunCharts(null, null);
        return;
      }
      var yearsInRun = comparison.entries.map(function(entry){ return entry.year; });
      if (!state.selectedYear || yearsInRun.indexOf(state.selectedYear) === -1){
        state.selectedYear = yearsInRun[0];
        if (yearSelect){
          setSelectedOption(yearSelect, String(state.selectedYear));
        }
      }
      populateGradeSelect(state.selectedYear);
      var entriesForYear = comparison.entries.filter(function(entry){ return entry.year === state.selectedYear; });
      if (!state.selectedGrade || !entriesForYear.some(function(entry){ return entry.grade_id === state.selectedGrade; })){
        state.selectedGrade = entriesForYear.length ? entriesForYear[0].grade_id : null;
        if (gradeSelect && state.selectedGrade){
          setSelectedOption(gradeSelect, state.selectedGrade);
        }
      }
      var selectedEntry = entriesForYear.find(function(entry){ return entry.grade_id === state.selectedGrade; }) || entriesForYear[0];
      if (selectedEntry){
        state.selectedGrade = selectedEntry.grade_id;
      }
      noteEl.textContent = selectedEntry ? ('Run ' + state.selectedRun + ' · ' + (selectedEntry.grade_label || selectedEntry.grade_id)) : '';
      renderRunTable(comparison);
      updateRunSummary(selectedEntry);
      updateRunCharts(comparison, selectedEntry);
    }

    function computeHeatmapValues(entry){
      if (!entry) return null;
      return buildHeatmapData([entry], entry.grade_label || entry.grade_id);
    }

    function renderCohortTable(entries){
      if (!cohortTableBody) return;
      cohortTableBody.innerHTML = '';
      if (!entries){
        return;
      }
      entries.forEach(function(entry){
        var tr = document.createElement('tr');
        tr.innerHTML = [
          '<td>' + entry.year + '</td>',
          '<td>' + (entry.grade_label || entry.grade_id) + '</td>',
          '<td>' + entry.run_count + '</td>',
          '<td>' + entry.sample_count + '</td>',
          '<td>' + formatPercent(entry.avg_llm_score_pct) + '</td>',
          '<td>' + formatPercent(entry.avg_human_percentile) + '</td>',
          '<td>' + formatPercent(entry.p25_percentile) + '</td>',
          '<td>' + formatPercent(entry.median_percentile) + '</td>',
          '<td>' + formatPercent(entry.p75_percentile) + '</td>',
          '<td>' + (entry.best_run_id || '–') + '</td>',
          '<td>' + (entry.worst_run_id || '–') + '</td>'
        ].join('');
        cohortTableBody.appendChild(tr);
      });
    }

    function updateCohortCharts(entries, markers){
      if (!entries || !entries.length){
        charts.updateCDF(cdfCanvas, [], [], 0);
        charts.updateHeatmap(heatmapCanvas, { xLabels: [], yLabels: [], values: [] });
        charts.updateBoxViolin(percentileCanvas, []);
        chartsSection.hidden = true;
        return;
      }
      var entry = entries[0];
      var year = entry.year;
      var gradeId = entry.grade_id;
      Promise.all([
        fetchYearSummary(year),
        fetchCdf(year, gradeId)
      ]).then(function(results){
        var cdf = results[1];
        charts.updateCDF(
          cdfCanvas,
          (cdf && cdf.points) ? cdf.points.map(function(point){ return { score: point.score, percentile: point.percentile }; }) : [],
          markers || [],
          entry.max_points
        );
        var heatmapData = buildHeatmapData(entries, entry.grade_label || entry.grade_id);
        charts.updateHeatmap(heatmapCanvas, heatmapData);

        var boxData = [];
        entries.forEach(function(item){
          if (item.median_percentile !== null && item.median_percentile !== undefined){
            boxData.push({
              label: item.grade_label || item.grade_id,
              stats: {
                min: item.min_percentile !== null && item.min_percentile !== undefined ? item.min_percentile : item.median_percentile,
                p25: item.p25_percentile !== null && item.p25_percentile !== undefined ? item.p25_percentile : item.median_percentile,
                median: item.median_percentile,
                p75: item.p75_percentile !== null && item.p75_percentile !== undefined ? item.p75_percentile : item.median_percentile,
                max: item.max_percentile !== null && item.max_percentile !== undefined ? item.max_percentile : item.median_percentile,
              }
            });
          }
        });
        charts.updateBoxViolin(percentileCanvas, boxData);
        chartsSection.hidden = false;
      }).catch(function(err){
        console.error('Failed to update cohort charts', err);
        chartsSection.hidden = false;
      });
    }

    function renderCohortView(){
      ensureCohortDefaults();
      var runIds = state.selectedRuns.slice();
      if (!runIds.length){
        cohortView.hidden = true;
        chartsSection.hidden = true;
        return;
      }
      cohortView.hidden = false;
      runMultiWrapper.hidden = false;
      cohortTypeWrapper.hidden = false;
      var key = runIds.slice().sort().join('|');
      fetchCohort(runIds).then(function(payload){
        if (!payload){
          renderCohortTable([]);
          chartsSection.hidden = true;
          return;
        }
        var cohortStats = (state.selectedCohortType === 'macro') ? payload.macro : payload.micro;
        var entries = cohortStats.entries || [];
        if (!entries.length){
          renderCohortTable([]);
          chartsSection.hidden = true;
          return;
        }
        if (!state.selectedYear || !entries.some(function(entry){ return entry.year === state.selectedYear; })){
          state.selectedYear = entries[0].year;
          if (yearSelect){
            setSelectedOption(yearSelect, String(state.selectedYear));
          }
          populateGradeSelect(state.selectedYear);
        }
        var filteredEntries = entries.filter(function(entry){ return entry.year === state.selectedYear; });
        if (!state.selectedGrade || !filteredEntries.some(function(entry){ return entry.grade_id === state.selectedGrade; })){
          state.selectedGrade = filteredEntries.length ? filteredEntries[0].grade_id : null;
          if (gradeSelect && state.selectedGrade){
            setSelectedOption(gradeSelect, state.selectedGrade);
          }
        }
        var selectedEntries = filteredEntries.filter(function(entry){ return entry.grade_id === state.selectedGrade; });
        noteEl.textContent = state.selectedGrade ? ('Cohort · ' + (selectedEntries[0] ? (selectedEntries[0].grade_label || selectedEntries[0].grade_id) : state.selectedGrade)) : '';
        renderCohortTable(filteredEntries);

        Promise.all(runIds.map(fetchRunComparison)).then(function(){
          var markers = [];
          runIds.forEach(function(runId){
            var comparison = state.runComparisons[runId];
            if (!comparison || !comparison.entries) return;
            var entry = comparison.entries.find(function(item){ return item.year === state.selectedYear && item.grade_id === state.selectedGrade; });
            if (entry && entry.human_percentile !== null && entry.human_percentile !== undefined){
              markers.push({
                label: labelForRun(state.runs.find(function(run){ return run.run_id === runId; })),
                score: entry.llm_total,
                percentile: entry.human_percentile,
              });
            }
          });
          updateCohortCharts(selectedEntries, markers);
        });
      });
    }

    function updateView(){
      var isRunView = state.selectedView === 'run';
      if (runSelectWrapper) runSelectWrapper.hidden = !isRunView;
      if (runView) runView.hidden = !isRunView;
      if (runMultiWrapper) runMultiWrapper.hidden = isRunView;
      if (cohortTypeWrapper) cohortTypeWrapper.hidden = isRunView;
      if (cohortView) cohortView.hidden = isRunView;
      if (percentileCanvas) percentileCanvas.parentElement.parentElement.parentElement.style.display = isRunView ? 'none' : '';
      if (isRunView){
        if (!state.selectedRun && state.runs.length){
          state.selectedRun = state.runs[0].run_id;
          if (runSelect){
            setSelectedOption(runSelect, state.selectedRun);
          }
        }
        if (state.selectedRun){
          fetchRunComparison(state.selectedRun).then(function(){ renderRunView(); });
        }
      } else {
        ensureCohortDefaults();
        renderCohortView();
      }
    }

    if (viewSelect){
      viewSelect.addEventListener('change', function(){
        state.selectedView = this.value;
        updateView();
      });
    }
    if (runSelect){
      runSelect.addEventListener('change', function(){
        state.selectedRun = this.value;
        updateView();
      });
    }
    if (runMulti){
      runMulti.addEventListener('change', function(){
        state.selectedRuns = getSelectedRuns();
        renderCohortView();
      });
    }
    if (yearSelect){
      yearSelect.addEventListener('change', function(){
        state.selectedYear = parseInt(this.value, 10);
        populateGradeSelect(state.selectedYear);
        if (state.selectedView === 'run'){
          renderRunView();
        } else {
          renderCohortView();
        }
      });
    }
    if (gradeSelect){
      gradeSelect.addEventListener('change', function(){
        state.selectedGrade = this.value;
        if (state.selectedView === 'run'){
          renderRunView();
        } else {
          renderCohortView();
        }
      });
    }
    if (cohortTypeSelect){
      cohortTypeSelect.addEventListener('change', function(){
        state.selectedCohortType = this.value;
        renderCohortView();
      });
    }

    populateRunSelectors();
    populateYearSelect();
    if (state.selectedYear){
      populateGradeSelect(state.selectedYear);
    }
    updateView();
  }

  document.addEventListener('DOMContentLoaded', function(){
    initThemeToggle();
    initOverview();
    initRunDetail();
    initCompare();
    initHumans();
    initAnalysis();
  });
})();
