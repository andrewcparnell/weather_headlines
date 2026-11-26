function normalise(str) {
  return (str || '')
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .trim();
}

export class LocationSearch {
  constructor(inputEl, listEl, onSelect) {
    this.inputEl = inputEl;
    this.listEl = listEl;
    this.onSelect = onSelect;
    this.gazetteer = [];
    this.matches = [];
    this.activeIndex = -1;
    this.abortCtrl = null;
    this.debounceId = null;

    this._bind();
  }

  async load(url = 'assets/gazetteer.json') {
    try {
      const resp = await fetch(url);
      if (!resp.ok) return;
      const raw = await resp.json();
      this.gazetteer = raw.map(item => ({
        ...item,
        _normName: normalise(item.name),
        _normSub: normalise(item.sub),
        _normAliases: (item.aliases || []).map(normalise),
      }));
    } catch {
      // Offline fallback stays graceful
    }
  }

  _searchLocal(query) {
    const q = normalise(query);
    if (!q) return [];
    const scored = [];

    for (const entry of this.gazetteer) {
      let rank = 99;
      if (entry._normName === q || entry._normAliases.includes(q)) {
        rank = entry.kind === 'Country' ? 0 : 1;
      } else if (entry._normName.startsWith(q) || entry._normAliases.some(a => a.startsWith(q))) {
        rank = entry.kind === 'Country' ? 2 : 3;
      } else if (entry._normName.includes(q) || entry._normSub.startsWith(q)) {
        rank = entry.kind === 'Country' ? 4 : 5;
      }
      if (rank < 99) {
        scored.push({ rank, entry });
      }
    }

    scored.sort((a, b) => a.rank - b.rank || a.entry.name.localeCompare(b.entry.name));
    return scored.slice(0, 8).map(s => s.entry);
  }

  async _searchRemote(query) {
    if (query.trim().length < 3) return [];
    this.abortCtrl?.abort();
    const ctrl = new AbortController();
    this.abortCtrl = ctrl;

    try {
      const url = `https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(query.trim())}&count=6&language=en&format=json`;
      const resp = await fetch(url, { signal: ctrl.signal });
      if (!resp.ok) return [];
      const data = await resp.json();
      return (data.results || []).map(r => {
        const isCountry = (r.feature_code || '').startsWith('PCL');
        return {
          name: r.name,
          kind: isCountry ? 'Country' : 'Place',
          sub: [r.admin1, r.country].filter(Boolean).join(', ') || r.country || '',
          lat: r.latitude,
          lon: r.longitude,
          distance: isCountry ? 1.85 : 1.46,
        };
      });
    } catch {
      return [];
    }
  }

  _render(items) {
    this.matches = items;
    this.activeIndex = items.length ? 0 : -1;
    this.listEl.replaceChildren();

    if (!items.length) {
      this.listEl.hidden = true;
      this.inputEl.setAttribute('aria-expanded', 'false');
      return;
    }

    items.forEach((item, idx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.role = 'option';
      btn.className = `search-item${idx === this.activeIndex ? ' highlighted' : ''}`;

      const left = document.createElement('span');
      left.textContent = item.name;

      const right = document.createElement('span');
      right.className = 'search-item-sub';
      right.textContent = item.kind === 'Country' ? 'Country' : item.sub;

      btn.append(left, right);
      btn.onclick = () => this._choose(item);
      this.listEl.appendChild(btn);
    });

    this.listEl.hidden = false;
    this.inputEl.setAttribute('aria-expanded', 'true');
  }

  _choose(item) {
    this.listEl.hidden = true;
    this.inputEl.setAttribute('aria-expanded', 'false');
    this.inputEl.value = item.name;
    this.inputEl.blur();
    this.onSelect?.(item);
  }

  _bind() {
    this.inputEl.addEventListener('input', () => {
      const val = this.inputEl.value;
      clearTimeout(this.debounceId);
      if (!val.trim()) {
        this._render([]);
        return;
      }
      const local = this._searchLocal(val);
      this._render(local);

      if (local.length < 5 && val.trim().length >= 3) {
        this.debounceId = setTimeout(async () => {
          const remote = await this._searchRemote(val);
          if (this.inputEl.value !== val) return;
          const merged = [...local];
          for (const r of remote) {
            const dup = merged.some(
              m => normalise(m.name) === normalise(r.name) && Math.hypot(m.lat - r.lat, m.lon - r.lon) < 1.2
            );
            if (!dup) merged.push(r);
          }
          this._render(merged.slice(0, 8));
        }, 200);
      }
    });

    this.inputEl.addEventListener('keydown', e => {
      if (this.listEl.hidden || !this.matches.length) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.activeIndex = (this.activeIndex + 1) % this.matches.length;
        this._updateHighlight();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.activeIndex = (this.activeIndex - 1 + this.matches.length) % this.matches.length;
        this._updateHighlight();
      } else if (e.key === 'Enter' && this.activeIndex >= 0) {
        e.preventDefault();
        this._choose(this.matches[this.activeIndex]);
      } else if (e.key === 'Escape') {
        this.listEl.hidden = true;
      }
    });

    document.addEventListener('click', e => {
      if (!this.inputEl.closest('.search-box')?.contains(e.target)) {
        this.listEl.hidden = true;
        this.inputEl.setAttribute('aria-expanded', 'false');
      }
    });

    document.addEventListener('keydown', e => {
      if (e.key === '/' && document.activeElement !== this.inputEl) {
        e.preventDefault();
        this.inputEl.focus();
        this.inputEl.select();
      }
    });
  }

  _updateHighlight() {
    [...this.listEl.children].forEach((el, idx) => {
      el.classList.toggle('highlighted', idx === this.activeIndex);
    });
  }
}
