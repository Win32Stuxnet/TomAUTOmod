// --- Tabs ---
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
});

// --- Toast ---
function toast(message, type = 'success') {
    const el = document.getElementById('toast');
    el.textContent = message;
    el.className = 'toast ' + type + ' show';
    setTimeout(() => el.classList.remove('show'), 3000);
}

// --- API helpers ---
async function api(method, path, body) {
    const opts = {
        method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (body) opts.body = JSON.stringify(body);
    const resp = await fetch('/api' + path, opts);
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || 'Request failed');
    }
    return resp.json();
}

// --- Config ---
async function loadConfig() {
    const cfg = await api('GET', `/guilds/${GUILD_ID}/config`);
    const form = document.getElementById('config-form');

    for (const [key, val] of Object.entries(cfg)) {
        const input = form.querySelector(`[name="${key}"]`);
        if (!input) continue;
        if (input.type === 'checkbox') {
            input.checked = !!val;
        } else {
            input.value = val ?? '';
        }
    }
}

document.getElementById('config-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = {};

    for (const input of form.querySelectorAll('input, textarea, select')) {
        if (!input.name) continue;
        if (input.type === 'checkbox') {
            data[input.name] = input.checked;
        } else if (input.type === 'number') {
            if (input.value) data[input.name] = parseInt(input.value, 10);
        } else {
            if (input.value) {
                // Convert channel IDs to integers
                if (input.name.endsWith('_id')) {
                    data[input.name] = parseInt(input.value, 10);
                } else {
                    data[input.name] = input.value;
                }
            }
        }
    }

    try {
        await api('PATCH', `/guilds/${GUILD_ID}/config`, data);
        toast('Configuration saved!');
    } catch (err) {
        toast(err.message, 'error');
    }
});

// --- Filters ---
async function loadFilters() {
    const rules = await api('GET', `/guilds/${GUILD_ID}/filters`);
    const tbody = document.querySelector('#filters-table tbody');
    tbody.innerHTML = '';

    for (const rule of rules) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${esc(rule.rule_type)}</td>
            <td><code>${esc(rule.pattern)}</code></td>
            <td>${esc(rule.action)}</td>
            <td><button class="btn btn-danger btn-sm" data-pattern="${esc(rule.pattern)}">Delete</button></td>
        `;
        tr.querySelector('button').addEventListener('click', async () => {
            try {
                await api('DELETE', `/guilds/${GUILD_ID}/filters/${encodeURIComponent(rule.pattern)}`);
                toast('Filter removed.');
                loadFilters();
            } catch (err) {
                toast(err.message, 'error');
            }
        });
        tbody.appendChild(tr);
    }
}

document.getElementById('filter-add-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = {
        rule_type: form.rule_type.value,
        pattern: form.pattern.value,
        action: form.action.value,
    };
    try {
        await api('POST', `/guilds/${GUILD_ID}/filters`, data);
        form.pattern.value = '';
        toast('Filter added.');
        loadFilters();
    } catch (err) {
        toast(err.message, 'error');
    }
});

// --- Custom Commands ---
async function loadCommands() {
    const cmds = await api('GET', `/guilds/${GUILD_ID}/commands`);
    const tbody = document.querySelector('#commands-table tbody');
    tbody.innerHTML = '';

    for (const cmd of cmds) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><code>!${esc(cmd.name)}</code></td>
            <td>${esc(cmd.response)}</td>
            <td>${esc(cmd.description || '')}</td>
            <td><button class="btn btn-danger btn-sm" data-name="${esc(cmd.name)}">Delete</button></td>
        `;
        tr.querySelector('button').addEventListener('click', async () => {
            try {
                await api('DELETE', `/guilds/${GUILD_ID}/commands/${encodeURIComponent(cmd.name)}`);
                toast('Command removed.');
                loadCommands();
            } catch (err) {
                toast(err.message, 'error');
            }
        });
        tbody.appendChild(tr);
    }
}

document.getElementById('command-add-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const form = e.target;
    const data = {
        name: form.name.value,
        response: form.response.value,
        description: form.description.value || '',
    };
    try {
        await api('POST', `/guilds/${GUILD_ID}/commands`, data);
        form.name.value = '';
        form.response.value = '';
        form.description.value = '';
        toast('Command added.');
        loadCommands();
    } catch (err) {
        toast(err.message, 'error');
    }
});

// --- Utils ---
function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// --- Init ---
loadConfig();
loadFilters();
loadCommands();
