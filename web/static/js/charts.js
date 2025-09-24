(function(){
  if (!window.echarts){
    console.warn('ECharts library missing; charts initialization skipped.');
    window.DashboardCharts = function(){};
    window.DashboardCharts.prototype = {
      updateBreakdown: function(){},
      updateHistogram: function(){},
      updatePredicted: function(){},
      updateConfusion: function(){},
      updateCompareSeries: function(){},
      clear: function(){}
    };
    return;
  }

  var BASE_PALETTE = ['#3b82f6', '#6366f1', '#f97316', '#22c55e', '#ef4444', '#06b6d4', '#a855f7'];

  function palette(count){
    var colors = [];
    for (var i = 0; i < count; i++){
      colors.push(BASE_PALETTE[i % BASE_PALETTE.length]);
    }
    return colors;
  }

  function readVar(name, fallback){
    var styles = window.getComputedStyle(document.documentElement);
    var value = styles.getPropertyValue(name);
    return value ? value.trim() : fallback;
  }

  function currentTheme(){
    return {
      text: readVar('--text', '#1f2126'),
      muted: readVar('--muted', '#5b6170'),
      border: readVar('--border', '#d7d9e0'),
      bg: readVar('--bg', '#f7f7f9'),
      elevated: readVar('--bg-elevated', '#ffffff'),
    };
  }

  function percentValue(value){
    if (value === null || value === undefined || isNaN(value)){
      return 0;
    }
    return Math.round(Number(value) * 10) / 10;
  }

  function percentLabel(value){
    if (value === null || value === undefined || isNaN(value)){
      return '0%';
    }
    return percentValue(value).toFixed(1) + '%';
  }

  function axisPercent(value){
    if (typeof value !== 'number' || !isFinite(value)){
      return '';
    }
    return Math.round(value).toString() + '%';
  }

  function rotateForLabels(labels){
    if (!labels || !labels.length) return 0;
    var maxLen = labels.reduce(function(max, label){
      return Math.max(max, (label || '').length);
    }, 0);
    if (maxLen > 18) return 60;
    if (maxLen > 12) return 40;
    if (maxLen > 8) return 25;
    return 0;
  }

  function wrapLabel(label, maxChars){
    if (!label) return '';
    var limit = maxChars || 12;
    if (label.length <= limit) return label;
    var chunks = [];
    var text = String(label);
    while (text.length > 0){
      chunks.push(text.slice(0, limit));
      text = text.slice(limit);
    }
    return chunks.join('\n');
  }

  function DashboardCharts(){
    this.instances = {};
    var self = this;
    this._resizeObserver = window.ResizeObserver ? new ResizeObserver(function(entries){
      entries.forEach(function(entry){
        var inst = self._findInstanceByElement(entry.target);
        if (inst){
          inst.chart.resize();
        }
      });
    }) : null;
  }

  DashboardCharts.prototype._findInstanceByElement = function(element){
    for (var key in this.instances){
      if (Object.prototype.hasOwnProperty.call(this.instances, key)){
        var inst = this.instances[key];
        if (inst && inst.element === element){
          return inst;
        }
      }
    }
    return null;
  };

  DashboardCharts.prototype._dispose = function(key){
    var inst = this.instances[key];
    if (!inst) return;
    if (this._resizeObserver){
      this._resizeObserver.unobserve(inst.element);
    }
    inst.chart.dispose();
    delete this.instances[key];
  };

  DashboardCharts.prototype._chartFor = function(key, element){
    if (!element) return null;
    var inst = this.instances[key];
    if (inst && inst.element !== element){
      this._dispose(key);
      inst = null;
    }
    if (!inst){
      var chart = window.echarts.init(element, null, {
        renderer: 'canvas',
        useDirtyRect: true,
      });
      inst = { chart: chart, element: element };
      this.instances[key] = inst;
      if (this._resizeObserver){
        this._resizeObserver.observe(element);
      }
    }
    return inst.chart;
  };

  DashboardCharts.prototype._applyOption = function(key, element, option){
    var chart = this._chartFor(key, element);
    if (!chart) return;
    chart.clear();
    chart.setOption(option, { notMerge: true, lazyUpdate: true });
    if (typeof requestAnimationFrame === 'function'){
      requestAnimationFrame(function(){ chart.resize(); });
    } else {
      setTimeout(function(){ chart.resize(); }, 0);
    }
  };

  DashboardCharts.prototype._emptyOption = function(message){
    var theme = currentTheme();
    return {
      animation: false,
      title: {
        text: message || 'No data available',
        left: 'center',
        top: 'middle',
        textStyle: {
          color: theme.muted,
          fontSize: 14,
          fontWeight: 'normal',
        },
      },
      grid: { left: 0, right: 0, top: 0, bottom: 0 },
      xAxis: { show: false },
      yAxis: { show: false },
      series: [],
    };
  };

  DashboardCharts.prototype.updateBreakdown = function(element, key, breakdown){
    if (!element) return;
    var labels = Object.keys(breakdown || {});
    if (!labels.length){
      this._applyOption(key, element, this._emptyOption('No breakdown data'));
      return;
    }
    var values = labels.map(function(label){
      var entry = breakdown[label] || {};
      return percentValue((entry.accuracy || 0) * 100);
    });
    var theme = currentTheme();
    var option = {
      color: palette(values.length),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        valueFormatter: function(value){ return percentLabel(value); },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 16, containLabel: true },
      xAxis: {
        type: 'value',
        min: 0,
        max: 100,
        axisLabel: {
          color: theme.muted,
          formatter: axisPercent,
        },
        splitLine: {
          lineStyle: {
            color: theme.border,
            opacity: 0.35,
          }
        }
      },
      yAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          color: theme.text,
          interval: 0,
          hideOverlap: true,
          formatter: function(value){ return wrapLabel(value, 14); },
        },
        axisTick: { alignWithLabel: true }
      },
      series: [{
        type: 'bar',
        data: values,
        barMaxWidth: 28,
        label: {
          show: true,
          position: 'right',
          formatter: function(params){ return percentLabel(params.value); },
          color: theme.text,
          fontWeight: 600,
        },
        itemStyle: {
          borderRadius: [4, 4, 4, 4],
        },
      }],
    };
    this._applyOption(key, element, option);
  };

  DashboardCharts.prototype.updateHistogram = function(element, histogram){
    if (!element) return;
    if (!histogram || !histogram.counts || !histogram.counts.length){
      this._applyOption(element.id, element, this._emptyOption('No distribution data'));
      return;
    }
    var labels = [];
    for (var i = 0; i < histogram.counts.length; i++){
      var start = Math.round(histogram.bins[i] || 0);
      var end = Math.round(histogram.bins[i + 1] || start);
      labels.push(start + '–' + end);
    }
    var counts = histogram.counts.map(function(v){ return Number(v) || 0; });
    var theme = currentTheme();
    var key = element.id;
    var option = {
      color: [palette(1)[0]],
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: labels.length > 8 ? 60 : 30, containLabel: true },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: {
          color: theme.text,
          interval: 0,
          rotate: rotateForLabels(labels),
          hideOverlap: true,
          formatter: function(value){ return wrapLabel(value, 10); },
        },
        axisTick: { alignWithLabel: true },
        nameGap: 16,
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: theme.muted },
        splitLine: {
          lineStyle: { color: theme.border, opacity: 0.35 },
        },
      },
      series: [{
        type: 'bar',
        data: counts,
        barMaxWidth: 32,
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
        label: {
          show: true,
          position: 'top',
          color: theme.muted,
          formatter: '{c}',
        },
      }],
    };
    this._applyOption(key, element, option);
  };

  DashboardCharts.prototype.updatePredicted = function(element, counts){
    if (!element) return;
    var labels = Object.keys(counts || {});
    if (!labels.length){
      this._applyOption(element.id, element, this._emptyOption('No predictions recorded'));
      return;
    }
    var data = labels.map(function(label){
      return { name: label, value: counts[label] || 0 };
    });
    var theme = currentTheme();
    var option = {
      color: palette(labels.length),
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} ({d}%)',
      },
      legend: {
        bottom: 0,
        type: 'scroll',
        textStyle: { color: theme.muted },
      },
      series: [{
        type: 'pie',
        radius: ['45%', '70%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 6,
          borderColor: theme.elevated,
          borderWidth: 2,
        },
        label: {
          color: theme.text,
        },
        labelLine: {
          length: 18,
          length2: 12,
        },
        data: data,
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateConfusion = function(element, matrix){
    if (!element) return;
    var labels = Object.keys(matrix || {}).sort();
    if (!labels.length){
      this._applyOption(element.id, element, this._emptyOption('No confusion data'));
      return;
    }
    var data = [];
    var maxVal = 0;
    for (var r = 0; r < labels.length; r++){
      var rowKey = labels[r];
      var row = matrix[rowKey] || {};
      for (var c = 0; c < labels.length; c++){
        var colKey = labels[c];
        var value = Number(row[colKey] || 0);
        data.push([c, r, value]);
        if (value > maxVal){
          maxVal = value;
        }
      }
    }
    var theme = currentTheme();
    var gradient = [readVar('--bg-hover', '#eef2ff'), palette(1)[0]];
    var option = {
      tooltip: {
        position: 'top',
        formatter: function(params){
          var actual = labels[params.value[1]];
          var predicted = labels[params.value[0]];
          return 'Actual ' + actual + ' / Predicted ' + predicted + ': ' + params.value[2];
        },
      },
      grid: { left: '12%', right: '6%', top: 40, bottom: 60 },
      xAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: theme.text },
        splitArea: { show: true },
        splitLine: { show: true, lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      yAxis: {
        type: 'category',
        data: labels,
        axisLabel: { color: theme.text },
        splitArea: { show: true },
        splitLine: { show: true, lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      visualMap: {
        min: 0,
        max: maxVal || 1,
        calculable: false,
        orient: 'horizontal',
        left: 'center',
        bottom: 10,
        inRange: {
          color: gradient,
        },
        textStyle: { color: theme.muted },
      },
      series: [{
        name: 'Confusion',
        type: 'heatmap',
        data: data,
        label: {
          show: true,
          color: theme.text,
          formatter: function(params){
            return params.value[2] ? String(params.value[2]) : '';
          },
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.35)',
          },
        },
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.updateCompareSeries = function(element, seriesData){
    if (!element) return;
    var entries = (seriesData || []).filter(function(entry){ return entry && entry.metric; });
    if (!entries.length){
      this._applyOption(element.id, element, this._emptyOption('No comparison data'));
      return;
    }
    var categories = entries.map(function(entry){ return entry.metric; });
    var values = entries.map(function(entry){
      if (entry.delta === null || entry.delta === undefined || isNaN(entry.delta)){
        return 0;
      }
      return Math.round(Number(entry.delta) * 1000) / 1000;
    });
    var theme = currentTheme();
    var data = values.map(function(value){
      return {
        value: value,
        itemStyle: { color: value >= 0 ? palette(1)[0] : '#ef4444' },
        label: {
          position: value >= 0 ? 'top' : 'bottom'
        }
      };
    });
    var option = {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        valueFormatter: function(value){
          if (value === null || value === undefined) return '0';
          return Number(value).toFixed(3);
        },
      },
      grid: { left: '10%', right: '6%', top: 40, bottom: 40, containLabel: true },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: {
          color: theme.text,
          interval: 0,
          rotate: rotateForLabels(categories),
          hideOverlap: true,
          formatter: function(value){
            var normalised = (value || '').replace(/_/g, ' ');
            return wrapLabel(normalised, 12);
          },
        },
        axisTick: { alignWithLabel: true },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: theme.muted },
        splitLine: { lineStyle: { color: theme.border, opacity: 0.35 } },
      },
      series: [{
        type: 'bar',
        data: data,
        barMaxWidth: 32,
        label: {
          show: true,
          color: theme.text,
          formatter: function(params){
            return Number(params.value).toFixed(3);
          },
        },
        itemStyle: {
          borderRadius: [4, 4, 0, 0],
        },
        markLine: {
          silent: true,
          symbol: 'none',
          data: [{ yAxis: 0 }],
          lineStyle: {
            color: theme.border,
            type: 'dashed',
          }
        }
      }],
    };
    this._applyOption(element.id, element, option);
  };

  DashboardCharts.prototype.clear = function(){
    for (var key in this.instances){
      if (Object.prototype.hasOwnProperty.call(this.instances, key)){
        this._dispose(key);
      }
    }
    this.instances = {};
  };

  window.DashboardCharts = DashboardCharts;
})();
