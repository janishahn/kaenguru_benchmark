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
        cost_min: '',
        cost_max: '',
        page: 1,
        page_size: 25,
        sort_by: 'id',
        sort_dir: 'asc'
      }
    };

    var tableBody = $('#results-table tbody');
    var pagination = $('#results-pagination');
    var filterForm = $('#filters-form');
    var chipContainers = $$('.chip-select', filterForm);
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
      ['points_min','points_max','latency_min','latency_max','tokens_min','tokens_max','cost_min','cost_max'].forEach(function(field){
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
      ['points','latency','tokens','cost'].forEach(function(prefix){
        var minKey = prefix + '_min';
        var maxKey = prefix + '_max';
        if (state.filters[minKey] || state.filters[maxKey]){
          items.push(prefix + ': ' + (state.filters[minKey] || '–') + '…' + (state.filters[maxKey] || '–'));
        }
      });
      activeFilters.textContent = items.join(' · ');
    }

    function attachChipHandlers(){
      chipContainers.forEach(function(container){
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
            container.innerHTML = '';
            values.forEach(function(value){
              var btn = document.createElement('button');
              btn.type = 'button';
              btn.textContent = value;
              btn.dataset.value = value;
              container.appendChild(btn);
            });
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
        cell.colSpan = 13;
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

    function rerenderCharts(){
      if (!lastAggregates) return;
      charts.updateBreakdown($('#chart-group'), 'chart-group', lastAggregates.breakdown_by_group || {});
      charts.updateBreakdown($('#chart-year'), 'chart-year', lastAggregates.breakdown_by_year || {});
      charts.updateConfusion($('#chart-confusion'), lastAggregates.confusion_matrix || {});
      charts.updateHistogram($('#chart-latency'), lastAggregates.latency_hist || {});
      charts.updateHistogram($('#chart-token'), lastAggregates.tokens_hist || {});
      charts.updatePredicted($('#chart-predicted'), lastAggregates.predicted_counts || {});
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

  document.addEventListener('DOMContentLoaded', function(){
    initThemeToggle();
    initOverview();
    initRunDetail();
    initCompare();
  });
})();
