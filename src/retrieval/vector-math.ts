export function dot(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    let s = 0;
    for (let i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

export function norm(a: number[]): number {
    return Math.sqrt(dot(a, a));
}

export function unit(a: number[]): number[] {
    const m = norm(a);
    return m > 0 ? a.map((x) => x / m) : a.slice();
}

export function cosine(a: number[], b: number[]): number {
    const m = norm(a) * norm(b);
    return m > 0 ? dot(a, b) / m : 0;
}

/** Softmax with temperature; returns a uniform distribution when all inputs are equal. */
export function softmax(values: number[], temperature = 0.05): number[] {
    if (!values.length) return [];
    const t = temperature > 0 ? temperature : 0.05;
    const max = Math.max(...values);
    const exps = values.map((v) => Math.exp((v - max) / t));
    const sum = exps.reduce((s, x) => s + x, 0);
    return exps.map((x) => x / sum);
}

/**
 * Splits `total` slots in proportion to `weights` (largest-remainder method), never giving an
 * entry more than its `caps` allows. Slots an entry can't use pass to the others.
 */
export function allocate(total: number, weights: number[], caps: number[]): number[] {
    const n = weights.length;
    const out = new Array(n).fill(0);
    let remaining = Math.min(total, caps.reduce((s, c) => s + c, 0));
    let active = weights.map((w, i) => (caps[i] > 0 && w >= 0 ? i : -1)).filter((i) => i >= 0);

    while (remaining > 0 && active.length) {
        const wSum = active.reduce((s, i) => s + weights[i], 0);
        const share = (i: number) => (wSum > 0 ? weights[i] / wSum : 1 / active.length);
        const exact = active.map((i) => ({ i, x: remaining * share(i) }));
        let given = 0;
        exact.forEach(({ i, x }) => {
            const take = Math.min(Math.floor(x), caps[i] - out[i]);
            out[i] += take;
            given += take;
        });
        // Hand out the rest by largest remainder
        const order = exact
            .filter(({ i }) => out[i] < caps[i])
            .sort((a, b) => (b.x - Math.floor(b.x)) - (a.x - Math.floor(a.x)) || weights[b.i] - weights[a.i]);
        for (const { i } of order) {
            if (given >= remaining) break;
            out[i] += 1;
            given += 1;
        }
        remaining -= given;
        active = active.filter((i) => out[i] < caps[i]);
        if (given === 0) break;
    }
    return out;
}

/** Mean of 3 largest values (or fewer), null for an empty list. */
export function meanTop(values: number[], m = 3): number | null {
    if (!values.length) return null;
    const top = [...values].sort((a, b) => b - a).slice(0, m);
    return top.reduce((s, x) => s + x, 0) / top.length;
}
