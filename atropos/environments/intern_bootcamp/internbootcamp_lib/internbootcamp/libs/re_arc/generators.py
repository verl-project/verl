from .dsl import *
from .utils import *



def generate_dbc1a6ce(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(diff_lb, diff_ub, card_bounds)
    s = sample(inds, num)
    fgcol = choice(remove(bgc, colopts))
    gi = fill(c, fgcol, s)
    resh = frozenset()
    for x, r in enumerate(gi):
        if r.count(fgcol) > 1:
            resh = combine(resh, connect((x, r.index(fgcol)), (x, -1 + w - r[::-1].index(fgcol))))
    go = fill(c, 8, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgcol) > 1:
            resv = combine(resv, connect((x, r.index(fgcol)), (x, -1 + h - r[::-1].index(fgcol))))
    go = dmirror(fill(dmirror(go), 8, resv))
    go = fill(go, fgcol, s)
    return {'input': gi, 'output': go}


def generate_2281f1f4(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    card_h_bounds = (1, h // 2 + 1)
    card_w_bounds = (1, w // 2 + 1)
    numtop = unifint(diff_lb, diff_ub, card_w_bounds)
    numright = unifint(diff_lb, diff_ub, card_h_bounds)
    if numtop == numright == 1:
        numtop, numright = sample([1, 2], 2)
    tp = sample(interval(0, w - 1, 1), numtop)
    rp = sample(interval(1, h, 1), numright)
    res = combine(apply(lbind(astuple, 0), tp), apply(rbind(astuple, w - 1), rp))
    bgc = choice(colopts)
    dc = choice(remove(bgc, colopts))
    gi = fill(canvas(bgc, (h, w)), dc, res)
    go = fill(gi, 2, product(rp, tp))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_c1d99e64(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    nofrontcol = choice(colopts)
    noisefrontcol = choice(remove(nofrontcol, colopts))
    gi = canvas(nofrontcol, (h, w))
    cands = totuple(asindices(gi))
    horifront_bounds = (1, h//4)
    vertifront_bounds = (1, w//4)
    nhf = unifint(diff_lb, diff_ub, horifront_bounds)
    nvf = unifint(diff_lb, diff_ub, vertifront_bounds)
    vfs = mapply(compose(vfrontier, tojvec), sample(interval(0, w, 1), nvf))
    hfs = mapply(compose(hfrontier, toivec), sample(interval(0, h, 1), nhf))
    gi = fill(gi, noisefrontcol, combine(vfs, hfs))
    cands = totuple(ofcolor(gi, nofrontcol))
    kk = size(cands)
    midp = (h * w) // 2
    noise_bounds = (0, max(0, kk - midp - 1))
    num_noise = unifint(diff_lb, diff_ub, noise_bounds)
    noise = sample(cands, num_noise)
    gi = fill(gi, noisefrontcol, noise)
    go = fill(gi, 2, merge(colorfilter(frontiers(gi), noisefrontcol)))
    return {'input': gi, 'output': go}


def generate_623ea044(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    g = canvas(bgc, (h, w))
    fullinds = asindices(g)
    inds = totuple(asindices(g))
    card_bounds = (0, max(int(h * w * 0.1), 1))
    numdots = unifint(diff_lb, diff_ub, card_bounds)
    dots = sample(inds, numdots)
    gi = canvas(bgc, (h, w))
    fgc = choice(remove(bgc, colopts))
    gi = fill(gi, fgc, dots)
    go = fill(gi, fgc, mapply(rbind(shoot, UP_RIGHT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, DOWN_LEFT), dots))
    go = fill(go, fgc, mapply(rbind(shoot, UNITY), dots))
    go = fill(go, fgc, mapply(rbind(shoot, NEG_UNITY), dots))
    return {'input': gi, 'output': go}


def generate_1190e5a7(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    nhf_bounds = (1, h // 3)
    nvf_bounds = (1, w // 3)
    nhf = unifint(diff_lb, diff_ub, nhf_bounds)
    nvf = unifint(diff_lb, diff_ub, nvf_bounds)
    hf_options = interval(1, h - 1, 1)
    vf_options = interval(1, w - 1, 1)
    hf_selection = []
    for k in range(nhf):
        hf = choice(hf_options)
        hf_selection.append(hf)
        hf_options = difference(hf_options, (hf - 1, hf, hf + 1))
    vf_selection = []
    for k in range(nvf):
        vf = choice(vf_options)
        vf_selection.append(vf)
        vf_options = difference(vf_options, (vf - 1, vf, vf + 1))
    remcols = remove(bgc, colopts)
    rcf = lambda x: recolor(choice(remcols), x)
    hfs = mapply(chain(rcf, hfrontier, toivec), tuple(hf_selection))
    vfs = mapply(chain(rcf, vfrontier, tojvec), tuple(vf_selection))
    gi = paint(c, combine(hfs, vfs))
    go = canvas(bgc, (nhf + 1, nvf + 1))
    return {'input': gi, 'output': go}


def generate_5614dbcf(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 10)
    col_card_bounds = (1, 8)
    noise_card_bounds = (0, 8)
    colopts = remove(5, interval(1, 10, 1))
    noisedindscands = totuple(asindices(canvas(0, (3, 3))))
    d = unifint(diff_lb, diff_ub, dim_bounds)
    cells_card_bounds = (1, d * d)
    go = canvas(0, (d, d))
    inds = totuple(asindices(go))
    numocc = unifint(diff_lb, diff_ub, cells_card_bounds)
    numcol = unifint(diff_lb, diff_ub, col_card_bounds)
    occs = sample(inds, numocc)
    colset = sample(colopts, numcol)
    gi = upscale(go, THREE)
    for occ in inds:
        offset = multiply(3, occ)
        numnoise = unifint(diff_lb, diff_ub, noise_card_bounds)
        noise = sample(noisedindscands, numnoise)
        if occ in occs:
            col = choice(colset)
            go = fill(go, col, initset(occ))
            gi = fill(gi, col, shift(noisedindscands, offset))
        gi = fill(gi, 5, shift(noise, offset))
    return {'input': gi, 'output': go}


def generate_05269061(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (2, 30)
    colopts = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, dim_bounds)
    go = canvas(0, (d, d))
    gi = canvas(0, (d, d))
    if choice((True, False)):
        period_bounds = (2, min(2*d-2, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, 2*d-1, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:2*d-1]):
            lin = shoot(toivec(k), UP_RIGHT)
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    else:
        period_bounds = (2, min(d, 9))
        num = unifint(diff_lb, diff_ub, period_bounds)
        cols = tuple(choice(colopts) for k in range(num))
        keeps = [choice(interval(j, d, num)) for j in range(num)]
        for k, col in enumerate((cols * 30)[:d]):
            lin = hfrontier(toivec(k))
            go = fill(go, col, lin)
            if keeps[k % num] == k:
                gi = fill(gi, col, lin)
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


def generate_1c786137(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    num_cols_card_bounds = (1, 8)
    colopts = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    noise_card_bounds = (0, h * w)
    c = canvas(0, (h, w))
    inds = totuple(asindices(c))
    num_noise = unifint(diff_lb, diff_ub, noise_card_bounds)
    num_cols = unifint(diff_lb, diff_ub, num_cols_card_bounds)
    noiseinds = sample(inds, num_noise)
    colset = sample(colopts, num_cols)
    trgcol = choice(difference(colopts, colset))
    noise = frozenset((choice(colset), ij) for ij in noiseinds)
    gi = paint(c, noise)
    boxhrng = (3, max(3, h//2))
    boxwrng = (3, max(3, w//2))
    boxh = unifint(diff_lb, diff_ub, boxhrng)
    boxw = unifint(diff_lb, diff_ub, boxwrng)
    boxi = choice(interval(0, h - boxh + 1, 1))
    boxj = choice(interval(0, w - boxw + 1, 1))
    loc = (boxi, boxj)
    llc = add(loc, toivec(boxh - 1))
    urc = add(loc, tojvec(boxw - 1))
    lrc = add(loc, (boxh - 1, boxw - 1))
    l1 = connect(loc, llc)
    l2 = connect(loc, urc)
    l3 = connect(urc, lrc)
    l4 = connect(llc, lrc)
    l = l1 | l2 | l3 | l4
    gi = fill(gi, trgcol, l)
    go = crop(gi, increment(loc), (boxh - 2, boxw - 2))
    return {'input': gi, 'output': go}


def generate_2204b7a8(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (4, 30)
    colopts = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, dim_bounds)
        w = unifint(diff_lb, diff_ub, dim_bounds)
        bgc = choice(colopts)
        remcols = remove(bgc, colopts)
        c = canvas(bgc, (h, w))
        inds = totuple(shift(asindices(canvas(0, (h, w - 2))), RIGHT))
        ccol = choice(remcols)
        remcols2 = remove(ccol, remcols)
        c1 = choice(remcols2)
        c2 = choice(remove(c1, remcols2))
        nc_bounds = (1, (h * (w - 2)) // 2 - 1)
        nc = unifint(diff_lb, diff_ub, nc_bounds)
        locs = sample(inds, nc)
        if w % 2 == 1:
            locs = difference(locs, vfrontier(tojvec(w // 2)))
        gi = fill(c, c1, vfrontier(ORIGIN))
        gi = fill(gi, c2, vfrontier(tojvec(w - 1)))
        gi = fill(gi, ccol, locs)
        a = sfilter(locs, lambda ij: last(ij) < w // 2)
        b = difference(locs, a)
        go = fill(gi, c1, a)
        go = fill(go, c2, b)
        if len(palette(gi)) == 4:
            break
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_23581191(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(2, interval(0, 10, 1))
    f = fork(combine, hfrontier, vfrontier)
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgcol = choice(colopts)
    remcols = remove(bgcol, colopts)
    c = canvas(bgcol, (h, w))
    inds = totuple(asindices(c))
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    card_bounds = (1, (h * w) // 4)
    na = unifint(diff_lb, diff_ub, card_bounds)
    nb = unifint(diff_lb, diff_ub, card_bounds)
    a = sample(inds, na)
    b = sample(difference(inds, a), nb)
    gi = fill(c, acol, a)
    gi = fill(gi, bcol, b)
    fa = apply(first, a)
    la = apply(last, a)
    fb = apply(first, b)
    lb = apply(last, b)
    alins = sfilter(inds, lambda ij: first(ij) in fa or last(ij) in la)
    blins = sfilter(inds, lambda ij: first(ij) in fb or last(ij) in lb)
    go = fill(c, acol, alins)
    go = fill(go, bcol, blins)
    go = fill(go, 2, intersection(set(alins), set(blins)))
    go = fill(go, acol, a)
    go = fill(go, bcol, b)
    return {'input': gi, 'output': go}


def generate_8be77c9e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(cols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(gi, hmirror(gi))
    return {'input': gi, 'output': go}


def generate_6d0aefbc(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
    return {'input': gi, 'output': go}


def generate_74dd1130(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = dmirror(gi)
    return {'input': gi, 'output': go}


def generate_62c24649(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(
        hconcat(gi, vmirror(gi)),
        hconcat(hmirror(gi), hmirror(vmirror(gi)))
    )
    return {'input': gi, 'output': go}


def generate_6150a2bd(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot180(gi)
    return {'input': gi, 'output': go}


def generate_6fa7a44f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(gi, hmirror(gi))
    return {'input': gi, 'output': go}


def generate_8d5021e8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go1 = hconcat(vmirror(gi), gi)
    go2 = vconcat(go1, hmirror(go1))
    go = vconcat(hmirror(go1), go2)
    return {'input': gi, 'output': go}


def generate_0520fde7(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 2, set(a) & set(b))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_46442a0e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = h
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go1 = hconcat(gi, rot90(gi))
    go2 = hconcat(rot270(gi), rot180(gi))
    go = vconcat(go1, go2)
    return {'input': gi, 'output': go}


def generate_1b2d62fb(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(0, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 8, ofcolor(gia, 0) & ofcolor(gib, 0))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_3428a4f5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = 0
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remcols)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 3, (set(a) | set(b)) - (set(a) & set(b)))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_42a50994(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(diff_lb, diff_ub, card_bounds)
    numcols = unifint(diff_lb, diff_ub, (0, min(9, num)))
    inds = totuple(asindices(c))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    objs = objects(gi, F, T, T)
    res = merge(sizefilter(objs, 1))
    go = fill(gi, bgc, res)
    return {'input': gi, 'output': go}


def generate_08ed6ac7(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(difference(colopts, (1, 2, 3, 4)))
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    barrange = (4, w)
    locopts = interval(0, w, 1)
    nbars = unifint(diff_lb, diff_ub, barrange)
    barlocs = sample(locopts, nbars)
    barhopts = interval(0, h, 1)
    barhs = sample(barhopts, 4)
    barcols = [choice(remcols) for j in range(nbars)]
    barhsfx = [choice(barhs) for j in range(nbars - 4)] + list(barhs)
    shuffle(barhsfx)
    ordered = sorted(barhs)
    colord = interval(1, 5, 1)
    for col, (loci, locj) in zip(barcols, list(zip(barhsfx, barlocs))):
        bar = connect((loci, locj), (h - 1, locj))
        gi = fill(gi, col, bar)
        go = fill(go, colord[ordered.index(loci)], bar)
    return {'input': gi, 'output': go}


def generate_8f2ea7aa(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    d2 = d ** 2
    gi = canvas(bgc, (d2, d2))
    go = canvas(bgc, (d2, d2))
    minig = canvas(bgc, (d, d))
    inds = totuple(asindices(minig))
    mp = d2 // 2
    devrng = (0, mp)
    dev = unifint(diff_lb, diff_ub, devrng)
    devs = choice((+1, -1))
    num = mp + devs * dev
    num = max(min(num, d2), 0)
    locs = set(sample(inds, num))
    while shape(locs) != (d, d):
        locs.add(choice(totuple(set(inds) - locs)))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    cols = sample(remcols, ncols)
    for ij in locs:
        minig = fill(minig, choice(cols), {ij})
    itv = interval(0, d2, d)
    plcopts = totuple(product(itv, itv))
    plc = choice(plcopts)
    minigo = asobject(minig)
    gi = paint(gi, shift(minigo, plc))
    for ij in locs:
        go = paint(go, shift(minigo, multiply(ij, d)))
    return {'input': gi, 'output': go}


def generate_7fe24cdd(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = h
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go1 = hconcat(gi, rot90(gi))
    go2 = hconcat(rot270(gi), rot180(gi))
    go = vconcat(go1, go2)
    return {'input': gi, 'output': go}


def generate_85c4e7cd(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    go = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[::-1])):
        ulc = (idx, idx)
        lrc = (h*2 - 1 - idx, w*2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
        go = fill(go, co, bx)
    return {'input': gi, 'output': go}


def generate_8e5a5113(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 9))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    k = 4 if d < 7 else 3
    nbound = (2, k)
    num = unifint(diff_lb, diff_ub, nbound)
    rotfs = (identity, rot90, rot180, rot270)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    colbnds = (1, 8)
    ncols = unifint(diff_lb, diff_ub, colbnds)
    patcols = sample(remcols, ncols)
    bgcanv = canvas(bgc, (d, d))
    c = canvas(bgc, (d, d))
    inds = totuple(asindices(c))
    ncolbnds = (1, d ** 2 - 1)
    ncells = unifint(diff_lb, diff_ub, ncolbnds)
    indsss = sample(inds, ncells)
    for ij in indsss:
        c = fill(c, choice(patcols), {ij})
    barr = canvas(barc, (d, 1))
    fillinidx = choice(interval(0, num, 1))
    gi = rot90(rot270(c if fillinidx == 0 else bgcanv))
    go = rot90(rot270(c))
    for j in range(num - 1):
        c = rot90(c)
        gi = hconcat(hconcat(gi, barr), c if j + 1 == fillinidx else bgcanv)
        go = hconcat(hconcat(go, barr), c)
    if choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


def generate_4c4377d9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(cols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vconcat(hmirror(gi), gi)
    return {'input': gi, 'output': go}


def generate_a65b410d(diff_lb: float, diff_ub: float) -> dict:
    colopts = difference(interval(0, 10, 1), (1, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    mpi = h // 2
    mpj = w // 2
    devi = unifint(diff_lb, diff_ub, (0, mpi))
    devj = unifint(diff_lb, diff_ub, (0, mpj))
    if choice((True, False)):
        locj = devj
        loci = devi
    else:
        loci = h - devi
        locj = w - devj
    loci = max(min(h - 2, loci), 1)
    locj = max(min(w - 2, locj), 1)
    loc = (loci, locj)
    bgc = choice(colopts)
    linc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, connect((loci, 0), (loci, locj)))
    blues = shoot((loci + 1, locj - 1), (1, -1))
    f = lambda ij: connect(ij, (ij[0], 0)) if ij[1] >= 0 else frozenset({})
    blues = mapply(f, blues)
    greens = shoot((loci - 1, locj + 1), (-1, 1))
    greens = mapply(f, greens)
    go = fill(gi, 1, blues)
    go = fill(go, 3, greens)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_5168d44c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    doth = unifint(diff_lb, diff_ub, (1, h//3))
    dotw = unifint(diff_lb, diff_ub, (1, w//3))
    borderh = unifint(diff_lb, diff_ub, (1, h//4))
    borderw = unifint(diff_lb, diff_ub, (1, w//4))
    direc = choice((DOWN, RIGHT, UNITY))
    dotloci = randint(0, h - doth - 1 if direc == RIGHT else h - doth - borderh - 1)
    dotlocj = randint(0, w - dotw - 1 if direc == DOWN else w - dotw - borderw - 1)
    dotloc = (dotloci, dotlocj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dotcol = choice(remcols)
    remcols = remove(dotcol, remcols)
    boxcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    dotshap = (doth, dotw)
    starterdot = backdrop(frozenset({dotloc, add(dotloc, decrement(dotshap))}))
    bordershap = (borderh, borderw)
    offset = add(multiply(direc, dotshap), multiply(direc, bordershap))
    itv = interval(-15, 16, 1)
    itv = apply(lbind(multiply, offset), itv)
    dots = mapply(lbind(shift, starterdot), itv)
    gi = fill(gi, dotcol, dots)
    protobx = backdrop(frozenset({
        (dotloci - borderh, dotlocj - borderw),
        (dotloci + doth + borderh - 1, dotlocj + dotw + borderw - 1),
    }))
    bx = protobx - starterdot
    bxshifted = shift(bx, offset)
    go = fill(gi, boxcol, bxshifted)
    gi = fill(gi, boxcol, bx)
    return {'input': gi, 'output': go}


def generate_a9f96cdd(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 6, 7, 8))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    fgc = choice(remove(bgc, cols))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    for k in range(noccs):
        if len(locs) == 0:
            break
        loc = choice(totuple(locs))
        locs = locs - mapply(neighbors, neighbors(loc))
        plcd = {loc}
        gi = fill(gi, fgc, plcd)
        go = fill(go, 3, shift(plcd, (-1, -1)))
        go = fill(go, 7, shift(plcd, (1, 1)))
        go = fill(go, 8, shift(plcd, (1, -1)))
        go = fill(go, 6, shift(plcd, (-1, 1)))
    return {'input': gi, 'output': go}


def generate_9172f3a0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (1, 10))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = upscale(gi, 3)
    return {'input': gi, 'output': go}


def generate_67a423a3(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    lineh = unifint(diff_lb, diff_ub, (1, h // 3))
    linew = unifint(diff_lb, diff_ub, (1, w // 3))
    loci = randint(1, h - lineh - 1)
    locj = randint(1, w - linew - 1)
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    for a in range(lineh):
        gi = fill(gi, acol, connect((loci+a, 0), (loci+a, w-1)))
    for b in range(linew):
        gi = fill(gi, bcol, connect((0, locj+b), (h-1, locj+b)))
    bx = outbox(frozenset({(loci, locj), (loci + lineh - 1, locj + linew - 1)}))
    go = fill(gi, 4, bx)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_db3e9e38(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    barth = unifint(diff_lb, diff_ub, (1, max(1, w // 5)))
    loci = unifint(diff_lb, diff_ub, (1, h - 2))
    locj = randint(1, w - barth - 1)
    bar = backdrop(frozenset({(loci, locj), (0, locj + barth - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, fgc, bar)
    go = canvas(bgc, (h, w))
    for k in range(16):
        rsh = multiply(2 * k, (-1, barth))
        go = fill(go, fgc, shift(bar, rsh))
        lsh = multiply(2 * k, (-1, -barth))
        go = fill(go, fgc, shift(bar, lsh))
        rsh = multiply(2 * k + 1, (-1, barth))
        go = fill(go, 8, shift(bar, rsh))
        lsh = multiply(2 * k + 1, (-1, -barth))
        go = fill(go, 8, shift(bar, lsh))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_9dfd6313(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    dh = unifint(diff_lb, diff_ub, (1, 14))
    d = 2 * dh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    gi = canvas(bgc, (d, d))
    inds = asindices(gi)
    lni = randint(1, 4)
    if lni == 1:
        ln = connect((dh, 0), (dh, d - 1))
        mirrf = hmirror
        cands = sfilter(inds, lambda ij: ij[0] > dh)
    elif lni == 2:
        ln = connect((0, dh), (d - 1, dh))
        mirrf = vmirror
        cands = sfilter(inds, lambda ij: ij[1] > dh)
    elif lni == 3:
        ln = connect((0, 0), (d - 1, d - 1))
        mirrf = dmirror
        cands = sfilter(inds, lambda ij: ij[0] > ij[1])
    elif lni == 4:
        ln = connect((d - 1, 0), (0, d - 1))
        mirrf = cmirror
        cands = sfilter(inds, lambda ij: (ij[0] + ij[1]) > d)
    gi = fill(gi, linc, ln)
    mp = (d * (d - 1)) // 2
    numcols = unifint(diff_lb, diff_ub, (1, min(7, mp)))
    colsch = sample(remcols, numcols)
    numpix = unifint(diff_lb, diff_ub, (1, len(cands)))
    pixs = sample(totuple(cands), numpix)
    for pix in pixs:
        gi = fill(gi, choice(colsch), {pix})
    go = mirrf(gi)
    if choice((True, False)):
        gi, go = go, gi
    return {'input': gi, 'output': go}


def generate_746b3537(diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (1, 30))
    cols = []
    lastc = -1
    for k in range(h):
        c = choice(remove(lastc, fullcols))
        cols.append(c)
        lastc = c
    go = tuple((c,) for c in cols)
    gi = tuple(repeat(c, w) for c in cols)
    numinserts = unifint(diff_lb, diff_ub, (1, 30 - h))
    for k in range(numinserts):
        loc = randint(0, len(gi) - 1)
        gi = gi[:loc+1] + gi[loc:]
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_75b8110e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    canv = canvas(bgc, (h, w))
    cels = totuple(asindices(canv))
    mp = (h * w) // 2
    nums = []
    for k in range(4):
        dev = unifint(diff_lb, diff_ub, (0, mp))
        if choice((True, False)):
            num = h * w - dev
        else:
            num = dev
        num = min(max(0, num), h * w - 1)
        nums.append(num)
    s1, s2, s3, s4 = [sample(cels, num) for num in nums]
    gi1 = fill(canv, c1, s1)
    gi2 = fill(canv, c2, s2)
    gi3 = fill(canv, c3, s3)
    gi4 = fill(canv, c4, s4)
    gi = vconcat(hconcat(gi1, gi2), hconcat(gi3, gi4))
    go = fill(gi1, c4, s4)
    go = fill(go, c3, s3)
    go = fill(go, c2, s2)
    return {'input': gi, 'output': go}


def generate_1cf80156(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(colopts)
    fgc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    hb = unifint(diff_lb, diff_ub, (1, min(15, h - 1)))
    wb = unifint(diff_lb, diff_ub, (1, min(15, w - 1)))
    bounds = asindices(canvas(0, (hb, wb)))
    shp = {choice(totuple(corners(bounds)))}
    mp = (hb * wb) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    nc = choice((dev, hb * wb - dev))
    nc = max(0, min(hb * wb - 1, nc))
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    di = randint(0, h - height(shp))
    dj = randint(0, w - width(shp))
    shpp = shift(shp, (di, dj))
    gi = fill(gi, fgc, shpp)
    go = fill(canvas(bgc, shape(shp)), fgc, shp)
    return {'input': gi, 'output': go}


def generate_28bf18c6(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(colopts)
    fgc = choice(remove(bgc, colopts))
    gi = canvas(bgc, (h, w))
    hb = unifint(diff_lb, diff_ub, (1, min(14, h - 1)))
    wb = unifint(diff_lb, diff_ub, (1, min(14, w - 1)))
    bounds = asindices(canvas(0, (hb, wb)))
    shp = {choice(totuple(corners(bounds)))}
    mp = (hb * wb) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    nc = choice((dev, hb * wb - dev))
    nc = max(0, min(hb * wb - 1, nc))
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    di = randint(0, h - height(shp))
    dj = randint(0, w - width(shp))
    shpp = shift(shp, (di, dj))
    gi = fill(gi, fgc, shpp)
    go = fill(canvas(bgc, shape(shp)), fgc, shp)
    go = hconcat(go, go)
    return {'input': gi, 'output': go}


def generate_22eb0ac0(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    gi = canvas(0, (1, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(diff_lb, diff_ub, (1, h))
    locs = sample(interval(0, h, 1), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = sample(interval(0, h, 1), nlocs)
    mp = nlocs // 2
    nbarsdev = unifint(diff_lb, diff_ub, (0, mp))
    nbars = choice((nbarsdev, h - nbarsdev))
    nbars = max(0, min(nbars, nlocs))
    barlocs = sample(locs, nbars)
    nonbarlocs = difference(locs, barlocs)
    barcols = [choice(remcols) for j in range(nbars)]
    acols = [choice(remcols) for j in range(len(nonbarlocs))]
    bcols = [choice(remove(acols[j], remcols)) for j in range(len(nonbarlocs))]
    for bc, bl in zip(barcols, barlocs):
        gi = fill(gi, bc, ((bl, 0), (bl, w - 1)))
        go = fill(go, bc, connect((bl, 0), (bl, w - 1)))
    for (a, b), loc in zip(zip(acols, bcols), nonbarlocs):
        gi = fill(gi, a, {(loc, 0)})
        go = fill(go, a, {(loc, 0)})
        gi = fill(gi, b, {(loc, w - 1)})
        go = fill(go, b, {(loc, w - 1)})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_4258a5f9(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    mp = ((h * w) // 2) if (h * w) % 2 == 1 else ((h * w) // 2 - 1)
    ndots = unifint(diff_lb, diff_ub, (1, mp))
    inds = totuple(asindices(gi))
    dots = sample(inds, ndots)
    go = fill(gi, 1, mapply(neighbors, frozenset(dots)))
    go = fill(go, fgc, dots)
    gi = fill(gi, fgc, dots)
    return {'input': gi, 'output': go}


def generate_1e0a9b12(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    ff = chain(dmirror, lbind(apply, rbind(order, identity)), dmirror)
    while True:
        h = unifint(diff_lb, diff_ub, (3, 30))
        w = unifint(diff_lb, diff_ub, (3, 30))
        nc = unifint(diff_lb, diff_ub, (1, w))
        bgc = choice(cols)
        gi = canvas(bgc, (h, w))
        remcols = remove(bgc, cols)
        scols = [choice(remcols) for j in range(nc)]
        slocs = sample(interval(0, w, 1), nc)
        inds = totuple(connect(ORIGIN, (h - 1, 0)))
        for c, l in zip(scols, slocs):
            nc2 = randint(1, h - 1)
            sel = sample(inds, nc2)
            gi = fill(gi, c, shift(sel, tojvec(l)))
        go = replace(ff(replace(gi, bgc, -1)), -1, bgc)
        if colorcount(gi, bgc) > argmax(remove(bgc, palette(gi)), lbind(colorcount, gi)):
            break
    return {'input': gi, 'output': go}


def generate_9565186b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    wg = canvas(5, (h, w))
    numcols = unifint(diff_lb, diff_ub, (2, min(h * w - 1, 8)))
    mostcol = choice(cols)
    nummostcol_lb = (h * w) // numcols + 1
    nummostcol_ub = h * w - numcols + 1
    ubmlb = nummostcol_ub - nummostcol_lb
    nmcdev = unifint(diff_lb, diff_ub, (0, ubmlb))
    nummostcol = nummostcol_ub - nmcdev
    nummostcol = min(max(nummostcol, nummostcol_lb), nummostcol_ub)
    inds = totuple(asindices(wg))
    mostcollocs = sample(inds, nummostcol)
    gi = fill(wg, mostcol, mostcollocs)
    go = fill(wg, mostcol, mostcollocs)
    remcols = remove(mostcol, cols)
    othcols = sample(remcols, numcols - 1)
    reminds = difference(inds, mostcollocs)
    bufferlocs = sample(reminds, numcols - 1)
    for c, l in zip(othcols, bufferlocs):
        gi = fill(gi, c, {l})
    reminds = difference(reminds, bufferlocs)
    colcounts = {c: 1 for c in othcols}
    for ij in reminds:
        if len(othcols) == 0:
            gi = fill(gi, mostcol, {ij})
            go = fill(go, mostcol, {ij})
        else:
            chc = choice(othcols)
            gi = fill(gi, chc, {ij})
            colcounts[chc] += 1
            if colcounts[chc] == nummostcol - 1:
                othcols = remove(chc, othcols)
    return {'input': gi, 'output': go}


def generate_6e02f1e3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (3, 30))
    c = canvas(0, (d, d))
    inds = list(asindices(c))
    shuffle(inds)
    num = d ** 2
    numcols = choice((1, 2, 3))
    chcols = sample(cols, numcols)
    if len(chcols) == 1:
        gi = canvas(chcols[0], (d, d))
        go = canvas(0, (d, d))
        go = fill(go, 5, connect((0, 0), (0, d - 1)))
    elif len(chcols) == 2:
        c1, c2 = chcols
        mp = (d ** 2) // 2
        nc1 = unifint(diff_lb, diff_ub, (1, mp))
        a = inds[:nc1]
        b = inds[nc1:]
        gi = fill(c, c1, a)
        gi = fill(gi, c2, b)
        go = fill(canvas(0, (d, d)), 5, connect((0, 0), (d - 1, d - 1)))
    elif len(chcols) == 3:
        c1, c2, c3 = chcols
        kk = d ** 2
        a = int(1/3 * kk)
        b = int(2/3 * kk)
        adev = unifint(diff_lb, diff_ub, (0, a - 1))
        bdev = unifint(diff_lb, diff_ub, (0, kk - b - 1))
        a -= adev
        b -= bdev
        x1, x2, x3 = inds[:a], inds[a:b], inds[b:]
        gi = fill(c, c1, x1)
        gi = fill(gi, c2, x2)
        gi = fill(gi, c3, x3)
        go = fill(canvas(0, (d, d)), 5, connect((d - 1, 0), (0, d - 1)))
    return {'input': gi, 'output': go}


def generate_2dc579da(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    dotc = choice(remcols)
    hdev = unifint(diff_lb, diff_ub, (0, (h - 2) // 2))
    lineh = choice((hdev, h - 2 - hdev))
    lineh = max(min(h - 2, lineh), 1)
    wdev = unifint(diff_lb, diff_ub, (0, (w - 2) // 2))
    linew = choice((wdev, w - 2 - wdev))
    linew = max(min(w - 2, linew), 1)
    locidev = unifint(diff_lb, diff_ub, (1, h // 2))
    loci = choice((h // 2 - locidev, h // 2 + locidev))
    loci = min(max(1, loci), h - lineh - 1)
    locjdev = unifint(diff_lb, diff_ub, (1, w // 2))
    locj = choice((w // 2 - locjdev, w // 2 + locjdev))
    locj = min(max(1, locj), w - linew - 1)
    gi = canvas(bgc, (h, w))
    for a in range(loci, loci + lineh):
        gi = fill(gi, linc, connect((a, 0), (a, w - 1)))
    for b in range(locj, locj + linew):
        gi = fill(gi, linc, connect((0, b), (h - 1, b)))
    doth = randint(1, loci)
    dotw = randint(1, locj)
    dotloci = randint(0, loci - doth)
    dotlocj = randint(0, locj - dotw)
    dot = backdrop(frozenset({(dotloci, dotlocj), (dotloci + doth - 1, dotlocj + dotw - 1)}))
    gi = fill(gi, dotc, dot)
    go = crop(gi, (0, 0), (loci, locj))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_2dee498d(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 10))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    gi = hconcat(go, hconcat(go, go))
    return {'input': gi, 'output': go}


def generate_508bd3b6(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (h, 30))
    barh = unifint(diff_lb, diff_ub, (1, h // 2))
    barloci = unifint(diff_lb, diff_ub, (2, h - barh))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barc = choice(remcols)
    remcols = remove(barc, remcols)
    linc = choice(remcols)
    gi = canvas(bgc, (h, w))
    for j in range(barloci, barloci + barh):
        gi = fill(gi, barc, connect((j, 0), (j, w - 1)))
    dotlociinv = unifint(diff_lb, diff_ub, (0, barloci - 1))
    dotloci = min(max(0, barloci - 2 - dotlociinv), barloci - 1)
    ln1 = shoot((dotloci, 0), (1, 1))
    ofbgc = ofcolor(gi, bgc)
    ln1 = sfilter(ln1 & ofbgc, lambda ij: ij[0] < barloci)
    ln1 = order(ln1, first)
    ln2 = shoot(ln1[-1], (-1, 1))
    ln2 = sfilter(ln2 & ofbgc, lambda ij: ij[0] < barloci)
    ln2 = order(ln2, last)[1:]
    ln = ln1 + ln2
    k = len(ln1)
    lineleninv = unifint(diff_lb, diff_ub, (0, k - 2))
    linelen = k - lineleninv
    givenl = ln[:linelen]
    reml = ln[linelen:]
    gi = fill(gi, linc, givenl)
    go = fill(gi, 3, reml)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_88a62173(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    gib = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gib))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gib = fill(gib, col, chos)
        inds = difference(inds, chos)
    numchinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    numch = h * w - numchinv
    inds2 = totuple(asindices(gib))
    subs = sample(inds2, numch)
    go = hmirror(hmirror(gib))
    for x, y in subs:
        go = fill(go, choice(remove(go[x][y], colsch + [bgc])), {(x, y)})
    gi = canvas(bgc, (h*2+1, w*2+1))
    idxes = ((0, 0), (h+1, w+1), (h+1, 0), (0, w+1))
    trgloc = choice(idxes)
    remidxes = remove(trgloc, idxes)
    trgobj = asobject(go)
    otherobj = asobject(gib)
    gi = paint(gi, shift(trgobj, trgloc))
    for ij in remidxes:
        gi = paint(gi, shift(otherobj, ij))
    return {'input': gi, 'output': go}


def generate_3aa6fb7a(diff_lb: float, diff_ub: float) -> dict:
    base = (ORIGIN, RIGHT, DOWN, UNITY)
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    maxnum = ((h * w) // 2) // 3
    num = unifint(diff_lb, diff_ub, (1, maxnum))
    kk, tr = 0, 0
    maxtrials = num * 2
    binds = set()
    while kk < num and tr < maxtrials:
        loc = choice(inds)
        ooo = choice(base)
        oo = remove(ooo, base)
        oop = shift(oo, loc)
        if set(oop).issubset(inds):
            inds = difference(inds, totuple(combine(oop, totuple(mapply(dneighbors, oop)))))
            gi = fill(gi, fgc, oop)
            binds.add(add(ooo, loc))
            kk += 1
        tr += 1
    go = fill(gi, 1, binds)
    return {'input': gi, 'output': go}


def generate_3ac3eb23(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nlocs = unifint(diff_lb, diff_ub, (1, max(1, (w - 2) // 3)))
    locopts = interval(1, w - 1, 1)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for k in range(nlocs):
        if len(locopts) == 0:
            break
        locj = choice(locopts)
        locopts = difference(locopts, interval(locj - 2, locj + 3, 1))
        col = choice(remcols)
        gi = fill(gi, col, {(0, locj)})
        go = fill(go, col, {(p, locj) for p in interval(0, h, 2)})
        go = fill(go, col, {(p, locj - 1) for p in interval(1, h, 2)})
        go = fill(go, col, {(p, locj + 1) for p in interval(1, h, 2)})
    mf = choice((identity, rot90, rot180, rot270))
    gi = mf(gi)
    go = mf(go)
    return {'input': gi, 'output': go}


def generate_c3e719e8(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    gob = canvas(-1, (h**2, w**2))
    wg = canvas(-1, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    nmc = randint(max(1, (h * w) // (ncols + 1) + 1), h * w)
    inds = totuple(asindices(wg))
    mc = choice(cols)
    remcols = remove(mc, cols)
    mcc = sample(inds, nmc)
    inds = difference(inds, mcc)
    gi = fill(wg, mc, mcc)
    ocols = sample(remcols, ncols)
    k = len(inds) // ncols + 1
    for ocol in ocols:
        if len(inds) == 0:
            break
        ub = min(nmc - 1, len(inds))
        ub = min(ub, k)
        ub = max(ub, 1)
        locs = sample(inds, unifint(diff_lb, diff_ub, (1, ub)))
        inds = difference(inds, locs)
        gi = fill(gi, ocol, locs)
    gi = replace(gi, -1, mc)
    o = asobject(gi)
    gob = replace(gob, -1, 0)
    go = paint(gob, mapply(lbind(shift, o), apply(rbind(multiply, (h, w)), ofcolor(gi, mc))))
    return {'input': gi, 'output': go}


def generate_29c11459(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(5, interval(0, 10, 1))
    gi = canvas(0, (1, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 29))
    if w % 2 == 0:
        w = choice((max(5, w - 1), min(29, w + 1)))
    bgc = choice(colopts)
    remcols = remove(bgc, colopts)
    ncols = unifint(diff_lb, diff_ub, (2, len(remcols)))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(diff_lb, diff_ub, (1, h))
    locs = sample(interval(0, h, 1), nlocs)
    while set(locs).issubset({0, h - 1}):
        locs = sample(interval(0, h, 1), nlocs)
    acols = []
    bcols = []
    aforb = -1
    bforb = -1
    for k in range(nlocs):
        ac = choice(remove(aforb, ccols))
        acols.append(ac)
        aforb = ac
        bc = choice(remove(bforb, ccols))
        bcols.append(bc)
        bforb = bc
    for (a, b), loc in zip(zip(acols, bcols), sorted(locs)):
        gi = fill(gi, a, {(loc, 0)})
        gi = fill(gi, b, {(loc, w - 1)})
        go = fill(go, a, connect((loc, 0), (loc, w // 2 - 1)))
        go = fill(go, b, connect((loc, w // 2 + 1), (loc, w - 1)))
        go = fill(go, 5, {(loc, w // 2)})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_23b5c85d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    colopts = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (2, h - 1))
    ow = unifint(diff_lb, diff_ub, (2, w - 1))
    num = unifint(diff_lb, diff_ub, (1, 8))
    cnt = 0
    while cnt < num:
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        col = choice(colopts)
        colopts = remove(col, colopts)
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        gi2 = fill(gi, col, obj)
        if color(argmin(sfilter(partition(gi2), fork(equality, size, fork(multiply, height, width))), fork(multiply, height, width))) != col:
            break
        else:
            gi = gi2
            go = canvas(col, shape(obj))
        oh = unifint(diff_lb, diff_ub, (max(0, oh - 4), oh - 1))
        ow = unifint(diff_lb, diff_ub, (max(0, ow - 4), ow - 1))
        if oh < 1 or ow < 1:
            break
        cnt += 1
    return {'input': gi, 'output': go}


def generate_1bfc4729(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    if h % 2 == 1:
        h = choice((max(4, h - 1), min(30, h + 1)))
    alocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        alocj = max(min(w // 2, alocj - w // 2), 1)
    aloci = randint(1, h // 2 - 1)
    blocj = unifint(diff_lb, diff_ub, (w // 2, w - 1))
    if choice((True, False)):
        blocj = max(min(w // 2, blocj - w // 2), 1)
    bloci = randint(h // 2, h - 2)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    aloc = (aloci, alocj)
    bloc = (bloci, blocj)
    gi = fill(gi, acol, {aloc})
    gi = fill(gi, bcol, {bloc})
    go = fill(gi, acol, hfrontier(aloc))
    go = fill(go, bcol, hfrontier(bloc))
    go = fill(go, acol, connect((0, 0), (0, w - 1)))
    go = fill(go, bcol, connect((h - 1, 0), (h - 1, w - 1)))
    go = fill(go, acol, connect((0, 0), (h // 2 - 1, 0)))
    go = fill(go, acol, connect((0, w - 1), (h // 2 - 1, w - 1)))
    go = fill(go, bcol, connect((h // 2, 0), (h - 1, 0)))
    go = fill(go, bcol, connect((h // 2, w - 1), (h - 1, w - 1)))
    return {'input': gi, 'output': go}


def generate_47c1f68c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    objc = choice(remcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w - 1))
    bx = asindices(canv)
    obj = {choice(totuple(bx))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, obj)
        ch = choice(totuple(bx & dns))
        obj.add(ch)
        bx = bx - {ch}
    obj = recolor(objc, obj)
    gi = paint(canv, obj)
    gi1 = hconcat(hconcat(gi, canvas(linc, (h, 1))), canv)
    gi2 = hconcat(hconcat(canv, canvas(linc, (h, 1))), canv)
    gi = vconcat(vconcat(gi1, canvas(linc, (1, 2*w+1))), gi2)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    go = replace(go, objc, linc)
    scf = choice((identity, hmirror, vmirror, compose(hmirror, vmirror)))
    gi = scf(gi)
    go = scf(go)
    return {'input': gi, 'output': go}


def generate_178fcbfb(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    iforb = set()
    jforb = set()
    mp = (h * w) // 3
    for col in (2, 1, 3):
        bnd = unifint(diff_lb, diff_ub, (1, w if col == 2 else h // 2))
        for ndots in range(bnd):
            if col == 2:
                ij = choice(sfilter(inds, lambda ij: last(ij) not in jforb))
                jforb.add(last(ij))
            if col == 1 or col == 3:
                ij = choice(sfilter(inds, lambda ij: first(ij) not in iforb))
                iforb.add(first(ij))
            gi = fill(gi, col, initset(ij))
            go = fill(go, col, (vfrontier if col == 2 else hfrontier)(ij))
            inds = remove(ij, inds)
    return {'input': gi, 'output': go}


def generate_ae4f1146(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    dh = unifint(diff_lb, diff_ub, (2, h // 3))
    dw = unifint(diff_lb, diff_ub, (2, w // 3))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * dh * dw)))
    cards = interval(0, dh * dw, 1)
    ccards = sorted(sample(cards, min(num, len(cards))))
    sgs = []
    c1 = canvas(fgc, (dh, dw))
    inds = totuple(asindices(c1))
    for card in ccards:
        x = sample(inds, card)
        x1 = fill(c1, 1, x)
        sgs.append(asobject(x1))
    go = paint(c1, sgs[-1])
    gi = canvas(bgc, (h, w))
    inds2 = asindices(canvas(bgc, (h - dh, w - dw)))
    maxtr = 10
    for sg in sgs[::-1]:
        if len(inds2) == 0:
            break
        loc = choice(totuple(inds2))
        plcd = shift(sg, loc)
        tr = 0    
        while (not toindices(plcd).issubset(inds2)) and tr < maxtr:
            loc = choice(totuple(inds2))
            plcd = shift(sg, loc)
            tr += 1
        if tr < maxtr:
            inds2 = difference(inds2, toindices(plcd) | outbox(plcd))
            gi = paint(gi, plcd)
    return {'input': gi, 'output': go}


def generate_3de23699(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    c = canvas(bgc, (h, w))
    hi = unifint(diff_lb, diff_ub, (4, h))
    wi = unifint(diff_lb, diff_ub, (4, w))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    ncol = choice(remcols)
    tmpo = frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)})
    cnds = totuple(backdrop(inbox(tmpo)))
    mp = len(cnds) // 2
    dev = unifint(diff_lb, diff_ub, (0, mp))
    ncnds = choice((dev, len(cnds) - dev))
    ncnds = min(max(0, ncnds), len(cnds))
    ss = sample(cnds, ncnds)
    gi = fill(c, ccol, corners(tmpo))
    gi = fill(gi, ncol, ss)
    go = trim(crop(switch(gi, ccol, ncol), (loci, locj), (hi, wi)))
    return {'input': gi, 'output': go}


def generate_7ddcd7ec(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    crns = (((0, 0), (-1, -1)), ((0, 1), (-1, 1)), ((1, 0), (1, -1)), ((1, 1), (1, 1)))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (0, 4))
    chos = sample(crns, num)
    loci = randint(0, h - 2)
    locj = randint(0, w - 2)
    loc = (loci, locj)
    remcols = remove(bgc, cols)
    for sp, dr in crns:
        sp2 = add(loc, sp)
        col = choice(remcols)
        gi = fill(gi, col, {sp2})
        go = fill(go, col, {sp2})
        if (sp, dr) in chos:
            gi = fill(gi, col, {add(sp2, dr)})
            go = fill(go, col, shoot(sp2, dr))
    return {'input': gi, 'output': go}


def generate_5c2c9af4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    boxhd = unifint(diff_lb, diff_ub, (0, h // 2))
    boxwd = unifint(diff_lb, diff_ub, (0, w // 2))
    boxh = choice((boxhd, h - boxhd))
    boxw = choice((boxwd, w - boxwd))
    if boxh % 2 == 0:
        boxh = choice((boxh - 1, boxh + 1))
    if boxw % 2 == 0:
        boxw = choice((boxw - 1, boxw + 1))
    boxh = min(max(1, boxh), h if h % 2 == 1 else h - 1)
    boxw = min(max(1, boxw), w if w % 2 == 1 else w - 1)
    boxshap = (boxh, boxw)
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    loc = (loci, locj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    cpi = loci + boxh // 2
    cpj = locj + boxw // 2
    cp = (cpi, cpj)
    A = (loci, locj)
    B = (loci + boxh - 1, locj + boxw - 1)
    gi = fill(c, fgc, {A, B, cp})
    go = fill(c, fgc, {A, B, cp})
    cond = True
    ooo = {A, B, cp}
    if hline(ooo) and len(ooo) == 3:
        go = fill(go, fgc, hfrontier(cp))
        cond = False
    if vline(ooo) and len(ooo) == 3:
        go = fill(go, fgc, vfrontier(cp))
        cond = False
    k = 1
    while cond:
        f1 = k * (boxh // 2)
        f2 = k * (boxw // 2)
        ulci = cpi - f1
        ulcj = cpj - f2
        lrci = cpi + f1
        lrcj = cpj + f2
        ulc = (ulci, ulcj)
        lrc = (lrci, lrcj)
        bx = box(frozenset({ulc, lrc}))
        go2 = fill(go, fgc, bx)
        cond = go != go2
        go = go2
        k += 1
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_0b148d64(diff_lb: float, diff_ub: float) -> dict:
    itv = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(itv)
    remitv = remove(bgc, itv)
    g = canvas(bgc, (h, w))
    x = randint(3, h - 3)
    y = randint(3, w - 3)
    di = randint(2, h - x - 1)
    dj = randint(2, w - y - 1)
    A = backdrop(frozenset({(0, 0), (x, y)}))
    B = backdrop(frozenset({(x + di, 0), (h - 1, y)}))
    C = backdrop(frozenset({(0, y + dj), (x, w - 1)}))
    D = backdrop(frozenset({(x + di, y + dj), (h - 1, w - 1)}))
    cola = choice(remitv)
    colb = choice(remove(cola, remitv))
    trg = choice((A, B, C, D))
    rem = remove(trg, (A, B, C, D))
    subf = lambda bx: {
        choice(totuple(connect(ulcorner(bx), urcorner(bx)))),
        choice(totuple(connect(ulcorner(bx), llcorner(bx)))),
        choice(totuple(connect(urcorner(bx), lrcorner(bx)))),
        choice(totuple(connect(llcorner(bx), lrcorner(bx)))),
    }
    sampler = lambda bx: set(sample(
        totuple(bx),
        len(bx) - unifint(diff_lb, diff_ub, (0, len(bx) - 1))
    ))
    gi = fill(g, cola, sampler(trg) | subf(trg))
    for r in rem:
        gi = fill(gi, colb, sampler(r) | subf(r))
    go = subgrid(frozenset(trg), gi)
    return {'input': gi, 'output': go}


def generate_beb8660c(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (3, 30))
    h = unifint(diff_lb, diff_ub, (w, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    k = min(8, w - 1)
    k = unifint(diff_lb, diff_ub, (1, k))
    co = sample(remcols, k)
    wds = sorted(sample(interval(1, w, 1), k))
    for j, (c, l) in enumerate(zip(co, wds)):
        j = h - k - 1 + j
        gi = fill(gi, c, connect((j, 0), (j, l - 1)))
    gi = fill(gi, 8, connect((h - 1, 0), (h - 1, w - 1)))
    go = vmirror(gi)
    gi = list(list(r) for r in gi[:-1])
    shuffle(gi)
    gi = tuple(tuple(r) for r in gi)
    gi = gi + go[-1:]
    gif = tuple()
    for r in gi:
        nbc = r.count(bgc)
        ofs = randint(0, nbc)
        gif = gif + (r[-ofs:] + r[:-ofs],)
    gi = vmirror(gif)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_8d510a79(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    barloci = randint(2, h - 3)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    gi = canvas(bgc, (h, w))
    bar = connect((barloci, 0), (barloci, w - 1))
    gi = fill(gi, barcol, bar)
    go = tuple(e for e in gi)
    jinds = interval(0, w, 1)
    numtop = unifint(diff_lb, diff_ub, (1, w - 1))
    numbot = unifint(diff_lb, diff_ub, (1, w - 1))
    tops = sample(jinds, numtop)
    bots = sample(jinds, numbot)
    for t in tops:
        loci = randint(0, barloci - 2)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (0, t)))
        else:
            go = fill(go, col, connect(loc, (barloci - 1, t)))
    for t in bots:
        loci = randint(barloci + 2, h - 1)
        col = choice((1, 2))
        loc = (loci, t)
        gi = fill(gi, col, {loc})
        if col == 1:
            go = fill(go, col, connect(loc, (h - 1, t)))
        else:
            go = fill(go, col, connect(loc, (barloci + 1, t)))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_7468f01a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    sgc, fgc = sample(remcols, 2)
    oh = unifint(diff_lb, diff_ub, (2, max(2, int(h * (2/3)))))
    ow = unifint(diff_lb, diff_ub, (2, max(2, int(w * (2/3)))))
    gi = canvas(bgc, (h, w))
    go = canvas(sgc, (oh, ow))
    bounds = asindices(go)
    shp = {ORIGIN}
    nc = unifint(diff_lb, diff_ub, (0, max(1, (oh * ow) // 2)))
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(dneighbors, shp))))
    go = fill(go, fgc, shp)
    objx = asobject(vmirror(go))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    gi = paint(gi, shift(objx, (loci, locj)))
    return {'input': gi, 'output': go}


def generate_09629e4f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nrows, ncolumns = h, w
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    ncols = unifint(diff_lb, diff_ub, (2, min(7, (h * w) - 2)))
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    fullh, fullw = h * nrows + nrows - 1, w * ncolumns + ncolumns - 1
    gi = canvas(barcol, (fullh, fullw))
    locs = totuple(product(interval(0, fullh, h + 1), interval(0, fullw, w + 1)))
    trgloc = choice(locs)
    remlocs = remove(trgloc, locs)
    colssf = sample(remcols, ncols)
    colsss = remove(choice(colssf), colssf)
    trgssf = sample(inds, ncols - 1)
    gi = fill(gi, bgc, shift(inds, trgloc))
    for ij, cl in zip(trgssf, colsss):
        gi = fill(gi, cl, {add(trgloc, ij)})
    for rl in remlocs:
        trgss = sample(inds, ncols)
        tmpg = tuple(e for e in c)
        for ij, cl in zip(trgss, colssf):
            tmpg = fill(tmpg, cl, {ij})
        gi = paint(gi, shift(asobject(tmpg), rl))
    go = canvas(bgc, (fullh, fullw))
    go = fill(go, barcol, ofcolor(gi, barcol))
    for ij, cl in zip(trgssf, colsss):
        go = fill(go, cl, shift(inds, multiply(ij, (h+1, w+1))))
    return {'input': gi, 'output': go}


def generate_4347f46a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, col, box(obj))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_6d58a25d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    shp = normalize(frozenset({
    (0, 0), (1, 0), (1, 1), (1, -1), (2, -1), (2, -2), (2, 1), (2, 2), (3, 3), (3, -3)
    }))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    c1 = choice(remcols)
    c2 = choice(remove(c1, remcols))
    loci = randint(0, h - 4)
    locj = randint(0, w - 7)
    plcd = shift(shp, (loci, locj))
    rem = difference(inds, plcd)
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, len(rem) // 2 - 1)))
    nois = sample(rem, nnoise)
    gi = fill(c, c2, nois)
    gi = fill(gi, c1, plcd)
    ff = lambda ij: len(intersection(shoot(ij, (-1, 0)), plcd)) > 0
    trg = sfilter(nois, ff)
    gg = lambda ij: valmax(sfilter(plcd, lambda kl: kl[1] == ij[1]), first) + 1
    kk = lambda ij: connect((gg(ij), ij[1]), (h - 1, ij[1]))
    fullres = mapply(kk, trg)
    go = fill(gi, c2, fullres)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_363442ee(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 3))
    w = unifint(diff_lb, diff_ub, (1, 3))
    h = h * 2 + 1
    w = w * 2 + 1
    nremh = unifint(diff_lb, diff_ub, (2, 30 // h))
    nremw = unifint(diff_lb, diff_ub, (2, (30 - w - 1) // w))
    rsh = nremh * h
    rsw = nremw * w
    rss = (rsh, rsw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    rsi = canvas(bgc, rss)
    rso = canvas(bgc, rss)
    ls = canvas(bgc, ((nremh - 1) * h, w))
    ulc = canvas(bgc, (h, w))
    bar = canvas(barcol, (nremh * h, 1))
    dotcands = totuple(product(interval(0, rsh, h), interval(0, rsw, w)))
    dotcol = choice(remcols)
    dev = unifint(diff_lb, diff_ub, (1, len(dotcands) // 2))
    ndots = choice((dev, len(dotcands) - dev))
    ndots = min(max(1, ndots), len(dotcands))
    dots = sample(dotcands, ndots)
    nfullremcols = unifint(diff_lb, diff_ub, (1, 8))
    fullremcols = sample(remcols, nfullremcols)
    for ij in asindices(ulc):
        ulc = fill(ulc, choice(fullremcols), {ij})
    ulco = asobject(ulc)
    osf = (h//2, w//2)
    for d in dots:
        rsi = fill(rsi, dotcol, {add(osf, d)})
        rso = paint(rso, shift(ulco, d))
    gi = hconcat(hconcat(vconcat(ulc, ls), bar), rsi)
    go = hconcat(hconcat(vconcat(ulc, ls), bar), rso)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_855e0971(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nbarsd = unifint(diff_lb, diff_ub, (1, 4))
    nbars = choice((nbarsd, 11 - nbarsd))
    nbars = max(3, nbars)
    h = unifint(diff_lb, diff_ub, (nbars, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    barsizes = [2] * nbars
    while sum(barsizes) < h:
        j = randint(0, nbars - 1)
        barsizes[j] += 1
    gi = tuple()
    go = tuple()
    locs = interval(0, w, 1)
    dotc = choice(cols)
    remcols = remove(dotc, cols)
    lastcol = -1
    nloclbs = [choice((0, 1)) for k in range(len(barsizes))]
    if sum(nloclbs) < 2:
        loc1, loc2 = sample(interval(0, len(nloclbs), 1), 2)
        nloclbs[loc1] = 1
        nloclbs[loc2] = 1
    for bs, nloclb in zip(barsizes, nloclbs):
        col = choice(remove(lastcol, remcols))
        gim = canvas(col, (bs, w))
        gom = canvas(col, (bs, w))
        nl = unifint(diff_lb, diff_ub, (nloclb, w // 2))
        chlocs = sample(locs, nl)
        for jj in chlocs:
            idx = (randint(0, bs - 1), jj)
            gim = fill(gim, dotc, {idx})
            gom = fill(gom, dotc, vfrontier(idx))
        lastcol = col
        gi = gi + gim
        go = go + gom
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_137eaa0f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 4))
    w = unifint(diff_lb, diff_ub, (2, 4))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dotc = choice(remcols)
    remcols = remove(dotc, remcols)
    go = canvas(dotc, (h, w))
    inds = totuple(asindices(go))
    loc = choice(inds)
    reminds = remove(loc, inds)
    nc = unifint(diff_lb, diff_ub, (1, min(h * w - 1, 8)))
    choscols = sample(remcols, nc)
    cd = {c: set() for c in choscols}
    for c in choscols:
        ij = choice(reminds)
        cd[c].add(ij)
        reminds = remove(ij, reminds)
    for ri in reminds:
        cd[choice(choscols)].add(ri)
    for c, idxes in cd.items():
        go = fill(go, c, idxes)
    gih = unifint(diff_lb, diff_ub, (min(h, w) * 2, 30))
    giw = unifint(diff_lb, diff_ub, (min(h, w) * 2, 30))
    objs = tuple(
        normalize(insert((dotc, loc), frozenset({(c, ij) for ij in cd[c]}))) \
            for c in choscols
    )
    maxtr = min(h, w) * 2
    maxtrtot = 1000
    while True:
        succ = True
        gi = canvas(bgc, (gih, giw))
        inds = asindices(gi)
        for obj in objs:
            oh, ow = shape(obj)
            succ2 = False
            tr = 0
            while tr < maxtr and not succ2:
                loci = randint(0, gih - oh)
                locj = randint(0, giw - ow)
                plcd = shift(obj, (loci, locj))
                tr += 1
                if toindices(plcd).issubset(inds):
                    succ2 = True
            if succ2:
                gi = paint(gi, plcd)
                inds = difference(inds, toindices(plcd))
                inds = difference(inds, mapply(neighbors, toindices(plcd)))
            else:
                succ = False
                break
        if succ:
            break
        maxtrtot += 1
        if maxtrtot < 1000:
            break
        maxtr = int(maxtr * 1.5)
        gih = randint(gih, 30)
        giw = randint(giw, 30)
    return {'input': gi, 'output': go}


def generate_31aa019c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (5, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        canv = canvas(bgc, (h, w))
        inds = totuple(asindices(canv))
        mp = (h * w) // 2 - 1
        ncols = unifint(diff_lb, diff_ub, (2, min(9, mp // 2 - 1)))
        chcols = sample(cols, ncols)
        trgcol = chcols[0]
        chcols = chcols[1:]
        dic = {c: set() for c in chcols}
        nnoise = unifint(diff_lb, diff_ub, (2 * (ncols - 1), mp))
        locc = choice(inds)
        inds = remove(locc, inds)
        noise = sample(inds, nnoise)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for c in chcols:
            ij = choice(inds)
            dic[c].add(ij)
            inds = remove(ij, inds)
        for ij in noise:
            c = choice(chcols)
            dic[c].add(ij)
            inds = remove(ij, inds)
        gi = fill(canv, trgcol, {locc})
        for c, ss in dic.items():
            gi = fill(gi, c, ss)
        gi = fill(gi, trgcol, {locc})
        if len(sfilter(palette(gi), lambda c: colorcount(gi, c) == 1)) == 1:
            break
    lc = leastcolor(gi)
    locc = ofcolor(gi, lc)
    go = fill(canv, lc, locc)
    go = fill(go, 2, neighbors(first(locc)))
    return {'input': gi, 'output': go}


def generate_2bee17df(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    indord1 = apply(tojvec, interval(0, w, 1))
    indord2 = apply(rbind(astuple, w - 1), interval(1, h - 1, 1))
    indord3 = apply(lbind(astuple, h - 1), interval(w - 1, 0, -1))
    indord4 = apply(toivec, interval(h - 1, 0, -1))
    indord = indord1 + indord2 + indord3 + indord4
    k = len(indord)
    sp = randint(0, k)
    arr = indord[sp:] + indord[:sp]
    ep = randint(k // 2 - 3, k // 2 + 1)
    a = arr[:ep]
    b = arr[ep:]
    cola = choice(remcols)
    remcols = remove(cola, remcols)
    colb = choice(remcols)
    gi = fill(c, cola, a)
    gi = fill(gi, colb, b)
    nr = unifint(diff_lb, diff_ub, (1, min(4, min(h, w) // 2)))
    for kk in range(nr):
        ring = box(frozenset({(1 + kk, 1 + kk), (h - 1 - kk, w - 1 - kk)}))
        for br in (cola, colb):
            blacks = ofcolor(gi, br)
            bcands = totuple(ring & ofcolor(gi, bgc) & mapply(dneighbors, ofcolor(gi, br)))
            jj = len(bcands)
            jj2 = randint(max(0, jj // 2 - 2), min(jj, jj // 2 + 1))
            ss = sample(bcands, jj2)
            gi = fill(gi, br, ss)
    res = shift(merge(frontiers(trim(gi))), (1, 1))
    go = fill(gi, 3, res)
    return {'input': gi, 'output': go}


def generate_50cb2852(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 8))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 8, bd)
            go = fill(go, col, box(obj))
            box(obj)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_662c240a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 7))
    ng = unifint(diff_lb, diff_ub, (2, 30 // d))
    nc = unifint(diff_lb, diff_ub, (2, min(9, d ** 2)))
    c = canvas(-1, (d, d))
    inds = totuple(asindices(c))
    tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
    tcolset = sample(cols, nc)
    triaf = frozenset((choice(tcolset), ij) for ij in tria)
    triaf = triaf | dmirror(triaf)
    gik = paint(c, triaf)
    ndistinv = unifint(diff_lb, diff_ub, (0, (d * (d - 1) // 2 - 1)))
    ndist = d * (d - 1) // 2 - ndistinv
    distinds = sample(difference(inds, sfilter(inds, lambda ij: ij[0] == ij[1])), ndist)
    
    for ij in distinds:
        if gik[ij[0]][ij[1]] == gik[ij[1]][ij[0]]:
            gik = fill(gik, choice(remove(gik[ij[0]][ij[1]], tcolset)), {ij})
        else:
            gik = fill(gik, gik[ij[1]][ij[0]], {ij})
    gi = gik
    go = tuple(e for e in gik)
    concatf = choice((hconcat, vconcat))
    for k in range(ng - 1):
        tria = sfilter(inds, lambda ij: ij[1] >= ij[0])
        tcolset = sample(cols, nc)
        triaf = frozenset((choice(tcolset), ij) for ij in tria)
        triaf = triaf | dmirror(triaf)
        gik = paint(c, triaf)
        if choice((True, False)):
            gi = concatf(gi, gik)
        else:
            gi = concatf(gik, gi)
    return {'input': gi, 'output': go}


def generate_e8593010(diff_lb: float, diff_ub: float) -> dict:
    a = frozenset({frozenset({ORIGIN})})
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
    frozenset({ORIGIN, DOWN, UNITY}),
    frozenset({ORIGIN, DOWN, RIGHT}),
    frozenset({UNITY, DOWN, RIGHT}),
    frozenset({UNITY, ORIGIN, RIGHT}),
    shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
    shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    a, b, c = totuple(a), totuple(b), totuple(c)
    prs = [(a, 3), (b, 2), (c, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 2) // 2))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        objs, col = choice(prs)
        obj = choice(objs)
        while ntr < maxtr:
            if len(reminds) == 0:
                break
            loc = choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {'input': gi, 'output': go}


def generate_d9f24cd1(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    dotc = choice(remcols)
    locopts = interval(1, w - 1, 1)
    maxnloc = (w - 2) // 2
    nlins = unifint(diff_lb, diff_ub, (1, maxnloc))
    locs = []
    for k in range(nlins):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        locs.append(loc)
    ndots = unifint(diff_lb, diff_ub, (1, maxnloc))
    locopts = interval(1, w - 1, 1)
    dotlocs = []
    for k in range(ndots):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = remove(loc, locopts)
        locopts = remove(loc - 1, locopts)
        locopts = remove(loc + 1, locopts)
        dotlocs.append(loc)
    gi = canvas(bgc, (h, w))
    for l in locs:
        gi = fill(gi, linc, {(h - 1, l)})
    dotlocs2 = []
    for l in dotlocs:
        jj = randint(1, h - 2)
        gi = fill(gi, dotc, {(jj, l)})
        dotlocs2.append(jj)
    go = tuple(e for e in gi)
    for linloc in locs:
        if linloc in dotlocs:
            jj = dotlocs2[dotlocs.index(linloc)]
            go = fill(go, linc, connect((h - 1, linloc), (jj + 1, linloc)))
            go = fill(go, linc, connect((jj + 1, linloc + 1), (0, linloc + 1)))
        else:
            go = fill(go, linc, connect((h - 1, linloc), (0, linloc)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_90c28cc7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 10))
    w = unifint(diff_lb, diff_ub, (2, 10))
    nc = unifint(diff_lb, diff_ub, (2, 9))
    gi = canvas(-1, (h, w))
    inds = totuple(asindices(gi))
    colss = sample(cols, nc)
    for ij in inds:
        gi = fill(gi, choice(colss), {ij})
    gi = dmirror(dedupe(dmirror(dedupe(gi))))
    go = tuple(e for e in gi)
    h, w = shape(gi)
    fullh = unifint(diff_lb, diff_ub, (h, 30))
    fullw = unifint(diff_lb, diff_ub, (w, 30))
    inh = unifint(diff_lb, diff_ub, (h, fullh))
    inw = unifint(diff_lb, diff_ub, (w, fullw))
    while h < inh or w < inw:
        opts = []
        if h < inh:
            opts.append((h, identity))
        elif w < inw:
            opts.append((w, dmirror))
        dim, mirrf = choice(opts)
        idx = randint(0, dim - 1)
        gi = mirrf(gi)
        gi = gi[:idx+1] + gi[idx:]
        gi = mirrf(gi)
        h, w = shape(gi)
    while h < fullh or w < fullw:
        opts = []
        if h < fullh:
            opts.append(identity)
        elif w < fullw:
            opts.append(dmirror)
        mirrf = choice(opts)
        gi = mirrf(gi)
        gi = merge(tuple(sample((((0,) * width(gi),), gi), 2)))
        gi = mirrf(gi)
        h, w = shape(gi)
    return {'input': gi, 'output': go}


def generate_321b1fc6(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dmyc = choice(remcols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)
    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    go = fill(go, bgc, shpp)
    return {'input': gi, 'output': go}


def generate_6455b5f5(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 8))
    while True:
        h = unifint(diff_lb, diff_ub, (6, 30))
        w = unifint(diff_lb, diff_ub, (6, 30))
        bgc = choice(cols)
        fgc = choice(remove(bgc, cols))
        gi = canvas(bgc, (h, w))
        ub = int((h * w) ** 0.5 * 1.5)
        num = unifint(diff_lb, diff_ub, (1, ub))
        for k in range(num):
            objs = colorfilter(objects(gi, T, T, F), bgc)
            eligobjs = sfilter(objs, lambda o: height(o) > 2 or width(o) > 2)
            if len(eligobjs) == 0:
                break
            if choice((True, False)):
                ro = argmax(eligobjs, size)
            else:
                ro = choice(totuple(eligobjs))
            if choice((True, False)):
                vfr = height(ro) < width(ro)
            else:
                vfr = choice((True, False))
            if vfr and width(ro) < 3:
                vfr = False
            if (not vfr) and height(ro) < 3:
                vfr = True
            if vfr:
                j = randint(leftmost(ro)+1, rightmost(ro)-1)
                ln = connect((uppermost(ro), j), (lowermost(ro), j))
            else:
                j = randint(uppermost(ro)+1, lowermost(ro)-1)
                ln = connect((j, leftmost(ro)), (j, rightmost(ro)))
            gi = fill(gi, fgc, ln)
        objs = colorfilter(objects(gi, T, T, F), bgc)
        if valmin(objs, size) != valmax(objs, size):
            break
    lblues = mfilter(objs, matcher(size, valmin(objs, size)))
    dblues = mfilter(objs, matcher(size, valmax(objs, size)))
    go = fill(gi, 8, lblues)
    go = fill(go, 1, dblues)
    return {'input': gi, 'output': go}


def generate_4c5c2cf0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cc = choice(remcols)
    remcols = remove(cc, remcols)
    objc = choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 5 and shape(cells) == (3, 3):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_56ff96f3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(2, 7)
        ow = randint(2, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            if choice((True, False)):
                cnrs = ((loci, locj), (loci + oh - 1, locj + ow - 1))
            else:
                cnrs = ((loci + oh - 1, locj), (loci, locj + ow - 1))
            gi = fill(gi, col, cnrs)
            go = fill(go, col, bd)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_2c608aff(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    dcol = choice(remcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - backdrop(outbox(bd)))
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    gi = fill(gi, dcol, noise)
    go = tuple(e for e in gi)
    hs = interval(loci, loci + boxh, 1)
    ws = interval(locj, locj + boxw, 1)
    for ij in noise:
        a, b = ij
        if a in hs:
            go = fill(go, dcol, connect(ij, (a, locj)))
        elif b in ws:
            go = fill(go, dcol, connect(ij, (loci, b)))
    go = fill(go, ccol, bd)
    return {'input': gi, 'output': go}


def generate_e98196ab(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 14))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    topc = choice(remcols)
    remcols = remove(topc, remcols)
    botc = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    nocc = unifint(diff_lb, diff_ub, (2, (h * w) // 2))
    subs = sample(inds, nocc)
    numa = randint(1, nocc - 1)
    A = sample(subs, numa)
    B = difference(subs, A)
    topg = fill(c, topc, A)
    botg = fill(c, botc, B)
    go = fill(topg, botc, B)
    br = canvas(linc, (1, w))
    gi = vconcat(vconcat(topg, br), botg)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_c9f8e694(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    ncols = unifint(diff_lb, diff_ub, (1, min(h, 8)))
    nsq = unifint(diff_lb, diff_ub, (1, 8))
    gir = canvas(bgc, (h, w - 1))
    gil = tuple((choice(remcols),) for j in range(h))
    inds = asindices(gir)
    succ = 0
    fails = 0
    maxfails = nsq * 5
    while succ < nsq and fails < maxfails:
        loci = randint(0, h - 3)
        locj = randint(0, w - 3)
        lock = randint(loci+1, min(loci + max(1, 2*h//3), h - 1))
        locl = randint(locj+1, min(locj + max(1, 2*w//3), w - 1))
        bd = backdrop(frozenset({(loci, locj), (lock, locl)}))
        if bd.issubset(inds):
            gir = fill(gir, sqc, bd)
            succ += 1
            indss = inds - bd
        else:
            fails += 1
    locs = ofcolor(gir, sqc)
    gil = tuple(e if idx in apply(first, locs) else (bgc,) for idx, e in enumerate(gil))
    fullobj = toobject(locs, hupscale(gil, w))
    gi = hconcat(gil, gir)
    giro = paint(gir, fullobj)
    go = hconcat(gil, giro)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_eb5a1d5d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 10))
    go = canvas(-1, (d*2-1, d*2-1))
    colss = sample(cols, d)
    for j, cc in enumerate(colss):
        go = fill(go, cc, box(frozenset({(j, j), (2*d - 2 - j, 2*d - 2 - j)})))
    nvenl = unifint(diff_lb, diff_ub, (0, 30 - d))
    nhenl = unifint(diff_lb, diff_ub, (0, 30 - d))
    enl = [nvenl, nhenl]
    gi = tuple(e for e in go)
    while (enl[0] > 0 or enl[1] > 0) and max(shape(gi)) < 30:
        opts = []
        if enl[0] > 0:
            opts.append((identity, 0))
        if enl[1] > 0:
            opts.append((dmirror, 1))
        mirrf, ch = choice(opts)
        gi = mirrf(gi)
        idx = randint(0, len(gi) - 1)
        gi = gi[:idx+1] + gi[idx:]
        gi = mirrf(gi)
        enl[ch] -= 1
    return {'input': gi, 'output': go}


def generate_82819916(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ass, bss = sample(remcols, 2)
    itv = interval(0, w, 1)
    na = randint(2, w - 2)
    alocs = sample(itv, na)
    blocs = difference(itv, alocs)
    if min(alocs) > min(blocs):
        alocs, blocs = blocs, alocs
    llocs = randint(0, h - 1)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    numl = unifint(diff_lb, diff_ub, (1, max(1, (h-1)//2)))
    remlocs = remove(llocs, interval(0, h, 1))
    for k in range(numl):
        lloc = choice(remlocs)
        remlocs = remove(lloc, remlocs)
        a, b = sample(remcols, 2)
        gi = fill(gi, a, {(lloc, j) for j in alocs})
        gi = fill(gi, b, {(lloc, j) for j in blocs})
    cutoff = min(blocs) + 1
    go = tuple(e for e in gi)
    gi = fill(gi, bgc, backdrop(frozenset({(0, cutoff), (h - 1, w - 1)})))
    gi = fill(gi, ass, {(llocs, j) for j in alocs})
    gi = fill(gi, bss, {(llocs, j) for j in blocs})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_5daaa586(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    loci1 = randint(1, h - 4)
    locj1 = randint(1, w - 4)
    loci1dev = unifint(diff_lb, diff_ub, (0, loci1 - 1))
    locj1dev = unifint(diff_lb, diff_ub, (0, locj1 - 1))
    loci1 -= loci1dev
    locj1 -= locj1dev
    loci2 = unifint(diff_lb, diff_ub, (loci1 + 2, h - 2))
    locj2 = unifint(diff_lb, diff_ub, (locj1 + 2, w - 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c1, c2, c3, c4 = sample(remcols, 4)
    f1 = recolor(c1, hfrontier(toivec(loci1)))
    f2 = recolor(c2, hfrontier(toivec(loci2)))
    f3 = recolor(c3, vfrontier(tojvec(locj1)))
    f4 = recolor(c4, vfrontier(tojvec(locj2)))
    gi = canvas(bgc, (h, w))
    fronts = [f1, f2, f3, f4]
    shuffle(fronts)
    for fr in fronts:
        gi = paint(gi, fr)
    cands = totuple(ofcolor(gi, bgc))
    nn = len(cands)
    nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
    noise = sample(cands, nnoise)
    gi = fill(gi, c1, noise)
    while len(frontiers(gi)) > 4:
        gi = fill(gi, bgc, noise)
        nnoise = unifint(diff_lb, diff_ub, (1, max(1, nn // 3)))
        noise = sample(cands, nnoise)
        if len(set(noise) & ofcolor(gi, c1)) >= len(ofcolor(gi, bgc)):
            break
        gi = fill(gi, c1, noise)
    go = crop(gi, (loci1, locj1), (loci2 - loci1 + 1, locj2 - locj1 + 1))
    ns = ofcolor(go, c1)
    go = fill(go, c1, mapply(rbind(shoot, (-1, 0)), ns))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_68b16354(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hmirror(gi)
    return {'input': gi, 'output': go}


def generate_bb43febb(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 8))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, col, box(obj))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_9ecd008a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(diff_lb, diff_ub, (1, 2*h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2*h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2*h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_f25ffba3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    h = h * 2 + 1
    w = unifint(diff_lb, diff_ub, (3, 15))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 2))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    while uppermost(obj) > h // 2 - 1 or lowermost(obj) < h // 2 + 1:
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    gi = hconcat(gix, canv)
    go = hconcat(gix, vmirror(gix))
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    if choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def generate_3bdb4ada(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 8))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        if choice((True, False)):
            oh = 3
            ow = unifint(diff_lb, diff_ub, (1, max(1, w // 2 - 1))) * 2 + 1
        else:
            ow = 3
            oh = unifint(diff_lb, diff_ub, (1, max(1, h // 2 - 1))) * 2 + 1
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            if oh == 3:
                ln = {(loci + 1, j) for j in range(locj+1, locj+ow, 2)}
            else:
                ln = {(j, locj + 1) for j in range(loci+1, loci+oh, 2)}
            go = fill(go, bgc, ln)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_2013d3e2(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    gi1 = hconcat(gi, rot90(gi))
    gi2 = hconcat(rot270(gi), rot180(gi))
    gi = vconcat(gi1, gi2)
    fullh = unifint(diff_lb, diff_ub, (2*h, 30))
    fullw = unifint(diff_lb, diff_ub, (2*w, 30))
    gio = asobject(gi)
    gic = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - 2*h)
    locj = randint(0, fullw - 2*w)
    gi = paint(gic, shift(gio, (loci, locj)))
    reminds = difference(asindices(gi), ofcolor(gi, bgc))
    go = lefthalf(tophalf(subgrid(reminds, gi)))
    return {'input': gi, 'output': go}


def generate_aabf363d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 28))
    w = unifint(diff_lb, diff_ub, (3, 28))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cola = choice(remcols)
    remcols = remove(cola, remcols)
    colb = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    sp = choice(totuple(bounds))
    ub = min(h * w - 1, max(1, (2/3) * h * w))
    ncells = unifint(diff_lb, diff_ub, (1, ub))
    shp = {sp}
    for k in range(ncells):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    c2 = canvas(bgc, (h+2, w+2))
    gi = fill(c2, cola, shp)
    go = fill(c2, colb, shp)
    gi = fill(gi, colb, {choice(totuple(ofcolor(gi, bgc)))})
    return {'input': gi, 'output': go}


def generate_d037b0a7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nlocs = unifint(diff_lb, diff_ub, (1, w))
    locs = sample(interval(0, w, 1), nlocs)
    for j in locs:
        col = choice(remcols)
        loci = randint(0, h - 1)
        loc = (loci, j)
        gi = fill(gi, col, {loc})
        go = fill(go, col, connect(loc, (h - 1, j)))
    return {'input': gi, 'output': go}


def generate_e26a3af2(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nr = unifint(diff_lb, diff_ub, (1, 10))
    w = unifint(diff_lb, diff_ub, (4, 30))
    scols = sample(cols, nr)
    sgs = [canvas(col, (2, w)) for col in scols]
    numexp = unifint(diff_lb, diff_ub, (0, 30 - nr))
    for k in range(numexp):
        idx = randint(0, nr - 1)
        sgs[idx] = sgs[idx] + sgs[idx][-1:]
    sgs2 = []
    for idx, col in enumerate(scols):
        sg = sgs[idx]
        a, b = shape(sg)
        ub = (a * b) // 2 - 1
        nnoise = unifint(diff_lb, diff_ub, (0, ub))
        inds = totuple(asindices(sg))
        noise = sample(inds, nnoise)
        oc = remove(col, cols)
        noise = frozenset({(choice(oc), ij) for ij in noise})
        sg2 = paint(sg, noise)
        for idxx in [0, -1]:
            while sum([e == col for e in sg2[idxx]]) < w // 2:
                locs = [j for j, e in enumerate(sg2[idxx]) if e != col]
                ch = choice(locs)
                if idxx == 0:
                    sg2 = (sg2[0][:ch] + (col,) + sg2[0][ch+1:],) + sg2[1:]
                else:
                    sg2 = sg2[:-1] + (sg2[-1][:ch] + (col,) + sg2[-1][ch+1:],)
        sgs2.append(sg2)
    gi = tuple(row for sg in sgs2 for row in sg)
    go = tuple(row for sg in sgs for row in sg)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_b8825c91(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    go = tuple(e for e in gi)
    for alph in (2, 1):
        locidev = unifint(diff_lb, diff_ub, (1, alph*h))
        locjdev = unifint(diff_lb, diff_ub, (1, w))
        loci = alph*h - locidev
        locj = w - locjdev
        loci2 = unifint(diff_lb, diff_ub, (loci, alph*h - 1))
        locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
        bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
        gi = fill(gi, 4, bd)
        gi, go = rot180(gi), rot180(go)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_ba97ae07(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    lineh = unifint(diff_lb, diff_ub, (1, h // 3))
    linew = unifint(diff_lb, diff_ub, (1, w // 3))
    loci = randint(1, h - lineh - 1)
    locj = randint(1, w - linew - 1)
    acol = choice(remcols)
    bcol = choice(remove(acol, remcols))
    for a in range(lineh):
        gi = fill(gi, acol, connect((loci+a, 0), (loci+a, w-1)))
    for b in range(linew):
        gi = fill(gi, bcol, connect((0, locj+b), (h-1, locj+b)))
    for b in range(linew):
        go = fill(go, bcol, connect((0, locj+b), (h-1, locj+b)))
    for a in range(lineh):
        go = fill(go, acol, connect((loci+a, 0), (loci+a, w-1)))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_c909285e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nfronts = unifint(diff_lb, diff_ub, (1, (h + w) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    boxcol = choice(remcols)
    remcols = remove(boxcol, remcols)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    for k in range(nfronts):
        ff = choice((hfrontier, vfrontier))
        loc = choice(inds)
        inds = remove(loc, inds)
        col = choice(remcols)
        gi = fill(gi, col, ff(loc))
    oh = unifint(diff_lb, diff_ub, (3, max(3, (h - 2) // 2)))
    ow = unifint(diff_lb, diff_ub, (3, max(3, (w - 2) // 2)))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, boxcol, bx)
    go = subgrid(bx, gi)
    return {'input': gi, 'output': go}


def generate_d511f180(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (5, 8))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numbg = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    bginds = sample(inds, numbg)
    idx = randint(0, numbg)
    blues = bginds[:idx]
    greys = bginds[idx:]
    rem = difference(inds, bginds)
    gi = fill(c, 8, blues)
    gi = fill(gi, 5, greys)
    go = fill(c, 5, blues)
    go = fill(go, 8, greys)
    for ij in rem:
        col = choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


def generate_d0f5fe59(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nfound = 0
    trials = 0
    maxtrials = nobjs * 5
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while trials < maxtrials and nfound < nobjs:
        oh = unifint(diff_lb, diff_ub, (1, 5))
        ow = unifint(diff_lb, diff_ub, (1, 5))
        bx = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bx))
        shp = {sp}
        dev = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((dev, oh * ow - dev))
        ncells = min(max(1, ncells), oh * ow - 1)
        for k in range(ncells):
            ij = choice(totuple((bx - shp) & mapply(dneighbors, shp)))
            shp.add(ij)
        shp = normalize(shp)
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        plcd = shift(shp, loc)
        if plcd.issubset(inds):
            gi = fill(gi, fgc, plcd)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            nfound += 1
        trials += 1
    go = canvas(bgc, (nfound, nfound))
    go = fill(go, fgc, connect((0, 0), (nfound - 1, nfound - 1)))
    return {'input': gi, 'output': go}


def generate_6e82a1ae(diff_lb: float, diff_ub: float) -> dict:
    b = frozenset({frozenset({ORIGIN, RIGHT}), frozenset({ORIGIN, DOWN})})
    c = frozenset({
    frozenset({ORIGIN, DOWN, UNITY}),
    frozenset({ORIGIN, DOWN, RIGHT}),
    frozenset({UNITY, DOWN, RIGHT}),
    frozenset({UNITY, ORIGIN, RIGHT}),
    shift(frozenset({ORIGIN, UP, DOWN}), DOWN),
    shift(frozenset({ORIGIN, LEFT, RIGHT}), RIGHT)
    })
    d = set()
    for k in range(100):
        shp = {(0, 0)}
        for jj in range(3):
            shp.add(choice(totuple(mapply(dneighbors, shp) - shp)))
        shp = frozenset(normalize(shp))
        d.add(shp)
    d = frozenset(d)
    d, b, c = totuple(d), totuple(b), totuple(c)
    prs = [(b, 3), (c, 2), (d, 1)]
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    reminds = asindices(gi)
    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 2) // 3))
    maxtr = 10
    for k in range(nobjs):
        ntr = 0
        objs, col = choice(prs)
        obj = choice(objs)
        while ntr < maxtr:
            loc = choice(totuple(reminds))
            olcd = shift(obj, loc)
            if olcd.issubset(reminds):
                gi = fill(gi, fgc, olcd)
                go = fill(go, col, olcd)
                reminds = (reminds - olcd) - mapply(dneighbors, olcd)
                break
            ntr += 1
    return {'input': gi, 'output': go}


def generate_f2829549(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    go = fill(c, 3, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_ce22a75a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    ndots = unifint(diff_lb, diff_ub, (1, (h * w) // 3))
    dots = sample(totuple(asindices(c)), ndots)
    gi = fill(c, fgc, dots)
    go = fill(c, 1, mapply(neighbors, dots))
    go = fill(go, 1, dots)
    return {'input': gi, 'output': go}


def generate_3c9b0459(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot180(gi)
    return {'input': gi, 'output': go}


def generate_99b1bc43(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(bset) - set(aset)) | (set(aset) - set(bset))
    go = fill(c, 3, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_b6afb2da(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            go = fill(go, 2, bd)
            go = fill(go, 4, box(bd))
            go = fill(go, 1, corners(bd))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_c8f0f002(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(7, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    numo = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    orng = sample(inds, numo)
    rem = difference(inds, orng)
    gi = fill(c, 7, orng)
    go = fill(c, 5, orng)
    for ij in rem:
        col = choice(ccols)
        gi = fill(gi, col, {ij})
        go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


def generate_54d82841(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nshps = unifint(diff_lb, diff_ub, (1, w // 3))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locs = interval(1, w - 1, 1)
    for k in range(nshps):
        if len(locs) == 0:
            break
        loc = choice(locs)
        locs = remove(loc, locs)
        locs = remove(loc + 1, locs)
        locs = remove(loc - 1, locs)
        locs = remove(loc + 2, locs)
        locs = remove(loc - 2, locs)
        loci = randint(1, h - 1)
        col = choice(remcols)
        ij = (loci, loc)
        shp = neighbors(ij) - connect((loci + 1, loc - 1), (loci + 1, loc + 1))
        gi = fill(gi, col, shp)
        go = fill(go, col, shp)
        go = fill(go, 4, {(h - 1, loc)})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_d631b094(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = 0
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    nc = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 2 - 1)))
    c = canvas(bgc, (h, w))
    cands = totuple(asindices(c))
    cels = sample(cands, nc)
    gi = fill(c, fgc, cels)
    go = canvas(fgc, (1, nc))
    return {'input': gi, 'output': go}


def generate_7c008303(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 13))
    w = unifint(diff_lb, diff_ub, (2, 13))
    h = h * 2
    w = w * 2
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    fgc = choice(remcols)
    remcols = remove(fgc, remcols)
    fremcols = sample(remcols, unifint(diff_lb, diff_ub, (1, 4)))
    qc = [choice(fremcols) for j in range(4)]
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(0, nc), h * w)
    cels = sample(inds, nc)
    go = fill(c, fgc, cels)
    gi = canvas(bgc, (h + 3, w + 3))
    gi = paint(gi, shift(asobject(go), (3, 3)))
    gi = fill(gi, linc, connect((2, 0), (2, w + 2)))
    gi = fill(gi, linc, connect((0, 2), (h + 2, 2)))
    gi = fill(gi, qc[0], {(0, 0)})
    gi = fill(gi, qc[1], {(0, 1)})
    gi = fill(gi, qc[2], {(1, 0)})
    gi = fill(gi, qc[3], {(1, 1)})
    A = lefthalf(tophalf(go))
    B = righthalf(tophalf(go))
    C = lefthalf(bottomhalf(go))
    D = righthalf(bottomhalf(go))
    A2 = replace(A, fgc, qc[0])
    B2 = replace(B, fgc, qc[1])
    C2 = replace(C, fgc, qc[2])
    D2 = replace(D, fgc, qc[3])
    go = vconcat(hconcat(A2, B2), hconcat(C2, D2))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_dae9d2b5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    if len(set(aset) & set(bset)) == 0:
        bset = bset[:-1] + [choice(aset)]
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(aset) | set(bset)
    go = fill(c, 6, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_aedd82e4(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = 0
    remcols = remove(bgc, colopts)
    c = canvas(bgc, (h, w))
    card_bounds = (0, max(0, (h * w) // 2 - 1))
    num = unifint(diff_lb, diff_ub, card_bounds)
    numcols = unifint(diff_lb, diff_ub, (0, min(8, num)))
    inds = totuple(asindices(c))
    chosinds = sample(inds, num)
    choscols = sample(remcols, numcols)
    locs = interval(0, len(chosinds), 1)
    choslocs = sample(locs, numcols)
    gi = canvas(bgc, (h, w))
    for col, endidx in zip(choscols, sorted(choslocs)[::-1]):
        gi = fill(gi, col, chosinds[:endidx])
    objs = objects(gi, F, F, T)
    res = merge(sizefilter(objs, 1))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


def generate_c9e6f938(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
    return {'input': gi, 'output': go}


def generate_913fb3ed(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6, 8))
    sr = (2, 3, 8)
    tr = (1, 6, 4)
    prs = list(zip(sr, tr))
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 10)))
    inds = asindices(gi)
    for k in range(numc):
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        a, b = choice(prs)
        inds = (inds - neighbors(loc)) - outbox(neighbors(loc))
        inds = remove(loc, inds)
        gi = fill(gi, a, {loc})
        go = fill(go, a, {loc})
        go = fill(go, b, neighbors(loc))
    return {'input': gi, 'output': go}


def generate_6430c8c4(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    bar = canvas(linc, (h, 1))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(hconcat(A, bar), B)
    res = (set(inds) - set(aset)) - set(bset)
    go = fill(c, 3, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_c0f76784(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (6, 7, 8))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = choice((3, 4, 5))
        ow = oh
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            ccc = oh + 3
            bdx = backdrop(inbox(obj))
            gi = fill(gi, bgc, bdx)
            go = fill(go, ccc, bdx)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_3af2c5a8(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (1, 30)
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, vmirror(gi))
    go = vconcat(go, hmirror(go))
    return {'input': gi, 'output': go}


def generate_496994bd(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (2, h * w - 1))
    bx = asindices(canv)
    obj = {
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] < h//2)))),
        (choice(remcols), choice(totuple(sfilter(bx, lambda ij: ij[0] > h//2))))
    }
    for kk in range(nc - 2):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gix = paint(canv, obj)
    gix = apply(rbind(order, matcher(identity, bgc)), gix)
    flag = choice((True, False))
    gi = hconcat(gix, canv if flag else hconcat(canvas(bgc, (h, 1)), canv))
    go = hconcat(gix, vmirror(gix) if flag else hconcat(canvas(bgc, (h, 1)), vmirror(gix)))
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    if choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def generate_bd4472b8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 28))
    w = unifint(diff_lb, diff_ub, (2, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    ccols = sample(remcols, w)
    cc = (tuple(ccols),)
    br = canvas(linc, (1, w))
    lp = canvas(bgc, (h, w))
    gi = vconcat(vconcat(cc, br), lp)
    go = vconcat(vconcat(cc, br), lp)
    pt = hupscale(dmirror(cc), w)
    pto = asobject(pt)
    idx = 2
    while idx < h+3:
        go = paint(go, shift(pto, (idx, 0)))
        idx += w
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_fafffa47(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = set(inds) - (set(aset) | set(bset))
    go = fill(c, 2, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_67e8384a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    go = paint(canv, obj)
    go = hconcat(go, vmirror(go))
    go = vconcat(go, hmirror(go))
    return {'input': gi, 'output': go}


def generate_ed36ccf7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot270(gi)
    return {'input': gi, 'output': go}


def generate_67a3c6ac(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = vmirror(gi)
    return {'input': gi, 'output': go}


def generate_a416b8f3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 30))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = hconcat(gi, gi)
    return {'input': gi, 'output': go}


def generate_d10ecb37(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = crop(gi, (0, 0), (2, 2))
    return {'input': gi, 'output': go}


def generate_5bd6f4ac(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = rot90(crop(rot270(gi), (0, 0), (3, 3)))
    return {'input': gi, 'output': go}


def generate_7b7f7511(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 15))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, min(9, h * w - 1)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(go))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        go = fill(go, col, chos)
        inds = difference(inds, chos)
    if choice((True, False)):
        go = dmirror(go)
        gi = vconcat(go, go)
    else:
        gi = hconcat(go, go)
    return {'input': gi, 'output': go}


def generate_c59eb873(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 15))
    w = unifint(diff_lb, diff_ub, (1, 15))
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (0, min(9, h * w)))
    colsch = sample(remcols, numc)
    inds = totuple(asindices(gi))
    for col in colsch:
        num = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // numc)))
        chos = sample(inds, num)
        gi = fill(gi, col, chos)
        inds = difference(inds, chos)
    go = upscale(gi, 2)
    return {'input': gi, 'output': go}


def generate_b1948b0a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    npd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    np = choice((npd, h * w - npd))
    np = min(max(0, npd), h * w)
    gi = canvas(6, (h, w))
    inds = totuple(asindices(gi))
    pp = sample(inds, np)
    npp = difference(inds, pp)
    for ij in npp:
        gi = fill(gi, choice(cols), {ij})
    go = fill(gi, 2, pp)
    return {'input': gi, 'output': go}


def generate_25ff71a9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc-1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, fgc, shift(plcd, (1, 0)))
    return {'input': gi, 'output': go}


def generate_f25fbde4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, (min(15, h-1) * min(15, w-1)) // 2)))
    nc = choice((ncd, (h-1) * (w-1) - ncd))
    nc = min(max(1, ncd), (h-1) * (w-1) - 1)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (min(15, h - 1), min(15, w - 1))))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = compress(gi)
    go = upscale(go, 2)
    return {'input': gi, 'output': go}


def generate_a740d043(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    ncd = unifint(diff_lb, diff_ub, (1, max(1, ((h-1) * (w-1)) // 2)))
    nc = choice((ncd, (h-1) * (w-1) - ncd))
    nc = min(max(1, ncd), (h-1) * (w-1) - 1)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, len(remcols)))
    remcols = sample(remcols, numc)
    c = canvas(bgc, (h, w))
    bounds = asindices(canvas(-1, (h - 1, w - 1)))
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    obj = {(choice(remcols), ij) for ij in plcd}
    gi = paint(c, obj)
    go = compress(gi)
    go = replace(go, bgc, 0)
    return {'input': gi, 'output': go}


def generate_be94b721(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    no = unifint(diff_lb, diff_ub, (3, max(3, (h * w) // 16)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (no+1, max(no+1, 2*no)))
    inds = asindices(c)
    ch = choice(totuple(inds))
    shp = {ch}
    inds = remove(ch, inds)
    for k in range(nc - 1):
        shp.add(choice(totuple((inds - shp) & mapply(dneighbors, shp))))
    inds = (inds - shp) - mapply(neighbors, shp)
    trgc = choice(remcols)
    gi = fill(c, trgc, shp)
    go = fill(canvas(bgc, shape(shp)), trgc, normalize(shp))
    for k in range(no):
        if len(inds) == 0:
            break
        ch = choice(totuple(inds))
        shp = {ch}
        nc2 = unifint(diff_lb, diff_ub, (1, nc - 1))
        for k in range(nc2 - 1):
            cands = totuple((inds - shp) & mapply(dneighbors, shp))
            if len(cands) == 0:
                break
            shp.add(choice(cands))
        col = choice(remcols)
        gi = fill(gi, col, shp)
        inds = (inds - shp) - mapply(neighbors, shp)
    return {'input': gi, 'output': go}


def generate_44d8ac46(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        tr += 1
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(5, 7)
        ow = randint(5, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            ensuresq = choice((True, False))
            if ensuresq:
                dim = randint(1, min(oh, ow) - 2)
                iloci = randint(1, oh - dim - 1)
                ilocj = randint(1, ow - dim - 1)
                inpart = backdrop({(loci + iloci, locj + ilocj), (loci + iloci + dim - 1, locj + ilocj + dim - 1)})
            else:
                cnds = backdrop(inbox(bd))
                ch = choice(totuple(cnds))
                inpart = {ch}
                kk = unifint(diff_lb, diff_ub, (1, len(cnds)))
                for k in range(kk - 1):
                    inpart.add(choice(totuple((cnds - inpart) & mapply(dneighbors, inpart))))
            inpart = frozenset(inpart)
            hi, wi = shape(inpart)
            if hi == wi and len(inpart) == hi * wi:
                incol = 2
            else:
                incol = bgc
            gi = fill(gi, col, bd)
            go = fill(go, col, bd)
            gi = fill(gi, bgc, inpart)
            go = fill(go, incol, inpart)
            succ += 1
            indss = (indss - bd) - outbox(bd)
    return {'input': gi, 'output': go}


def generate_3618c87e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, linc, dotc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    ln = connect((0, 0), (0, w - 1))
    nlocs = unifint(diff_lb, diff_ub, (1, w//2))
    locs = []
    opts = interval(0, w, 1)
    for k in range(nlocs):
        if len(opts) == 0:
            break
        ch = choice(opts)
        locs.append(ch)
        opts = remove(ch, opts)
        opts = remove(ch-1, opts)
        opts = remove(ch+1, opts)
    nlocs = len(opts)
    gi = fill(c, linc, ln)
    go = fill(c, linc, ln)
    for j in locs:
        hh = randint(1, h - 3)
        lnx = connect((0, j), (hh, j))
        gi = fill(gi, linc, lnx)
        go = fill(go, linc, lnx)
        gi = fill(gi, dotc, {(hh+1, j)})
        go = fill(go, dotc, {(0, j)})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_b27ca6d3(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, dotc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    ndots = unifint(diff_lb, diff_ub, (0, (h * w) // 5))
    nbars = unifint(diff_lb, diff_ub, (0, (h * w) // 12))
    dot = frozenset({(dotc, (1, 1))}) | recolor(bgc, dneighbors((1, 1)))
    bar1 = fill(canvas(bgc, (4, 3)), dotc, {(1, 1), (2, 1)})
    bar2 = dmirror(bar1)
    bar1 = asobject(bar1)
    bar2 = asobject(bar2)
    opts = [dot] * ndots + [choice((bar1, bar2)) for k in range(nbars)]
    shuffle(opts)
    inds = shift(asindices(canvas(-1, (h+2, w+2))), (-1, -1))
    for elem in opts:
        loc = (-1, -1)
        tr = 0
        while not toindices(shift(elem, loc)).issubset(inds) and tr < 5:
            loc = choice(totuple(inds))
            tr += 1
        xx = shift(elem, loc)
        if toindices(xx).issubset(inds):
            gi = paint(gi, xx)
            if len(elem) == 12:
                go = paint(go, {cel if cel[0] != bgc else (3, cel[1]) for cel in xx})
            else:
                go = paint(go, xx)
            inds = inds - toindices(xx)
    return {'input': gi, 'output': go}


def generate_46f33fce(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 7))
    w = unifint(diff_lb, diff_ub, (2, 7))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h*2, w*2))
    inds = totuple(asindices(go))
    locs = sample(inds, nc)
    objo = frozenset({(choice(remcols), ij) for ij in locs})
    f = lambda cij: (cij[0], double(cij[1]))
    obji = shift(apply(f, objo), (1, 1))
    gi = paint(gi, obji)
    go = paint(go, objo)
    go = upscale(go, 4)
    return {'input': gi, 'output': go}


def generate_a79310a0(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, 2, shift(plcd, (1, 0)))
    return {'input': gi, 'output': go}


def generate_dc1df850(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (0, (h * w) // 2 - 1))
    nreddev = unifint(diff_lb, diff_ub, (0, nc // 2))
    nred = choice((nreddev, nc - nreddev))
    nred = min(max(0, nred), nc)
    inds = totuple(asindices(c))
    occ = sample(inds, nc)
    reds = sample(occ, nred)
    others = difference(occ, reds)
    c = fill(c, 2, reds)
    obj = frozenset({(choice(remcols), ij) for ij in others})
    c = paint(c, obj)
    gi = tuple(r for r in c)
    go = underfill(c, 1, mapply(neighbors, frozenset(reds)))
    return {'input': gi, 'output': go}


def generate_f76d97a5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(5, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    col = choice(cols)
    gi = canvas(5, (h, w))
    go = canvas(col, (h, w))
    numdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    num = choice((numdev, h * w - numdev))
    num = min(max(1, num), h * w)
    inds = totuple(asindices(gi))
    locs = sample(inds, num)
    gi = fill(gi, col, locs)
    go = fill(go, 0, locs)
    return {'input': gi, 'output': go}


def generate_0d3d703e(diff_lb: float, diff_ub: float) -> dict:
    incols = (1, 2, 3, 4, 5, 6, 8, 9)
    outcols = (5, 6, 4, 3, 1, 2, 9, 8)
    k = len(incols)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gi = canvas(-1, (h, w))
    go = canvas(-1, (h, w))
    inds = asindices(gi)
    numc = unifint(diff_lb, diff_ub, (1, k))
    idxes = sample(interval(0, k, 1), numc)
    for ij in inds:
        idx = choice(idxes)
        gi = fill(gi, incols[idx], {ij})
        go = fill(go, outcols[idx], {ij})
    return {'input': gi, 'output': go}


def generate_445eab21(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    bigcol, area = 0, 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        if oh * ow == area:
            continue
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, box(bd))
            succ += 1
            indss = indss - bd
            if oh * ow > area:
                bigcol, area = col, oh * ow
        tr += 1
    go = canvas(bigcol, (2, 2))
    return {'input': gi, 'output': go}


def generate_b94a9452(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, outer, inner = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    oh2d = unifint(diff_lb, diff_ub, (0, oh // 2))
    ow2d = unifint(diff_lb, diff_ub, (0, ow // 2))
    oh2 = choice((oh2d, oh - oh2d))
    oh2 = min(max(1, oh2), oh - 2)
    ow2 = choice((ow2d, ow - ow2d))
    ow2 = min(max(1, ow2), ow - 2)
    loci2 = randint(loci+1, loci+oh-oh2-1)
    locj2 = randint(locj+1, locj+ow-ow2-1)
    obj1 = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    obj2 = backdrop(frozenset({(loci2, locj2), (loci2 + oh2 - 1, locj2 + ow2 - 1)}))
    gi = fill(c, outer, obj1)
    gi = fill(gi, inner, obj2)
    go = compress(gi)
    go = switch(go, outer, inner)
    return {'input': gi, 'output': go}


def generate_e9afcf9a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(10, h)))
    colss = sample(cols, numc)
    rr = tuple(choice(colss) for k in range(h))
    rr2 = rr[::-1]
    gi = []
    go = []
    for k in range(w):
        gi.append(rr)
        if k % 2 == 0:
            go.append(rr)
        else:
            go.append(rr2)
    gi = dmirror(tuple(gi))
    go = dmirror(tuple(go))
    return {'input': gi, 'output': go}


def generate_e9614598(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    r = randint(0, h - 1)
    sizh = unifint(diff_lb, diff_ub, (2, w//2))
    siz = 2 * sizh + 1
    siz = min(max(5, siz), w)
    locj = randint(0, w - siz)
    bgc, dotc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    A = (r, locj)
    B = (r, locj+siz-1)
    gi = fill(c, dotc, {A, B})
    locc = (r, locj + siz // 2)
    go = fill(gi, 3, {locc})
    go = fill(go, 3, dneighbors(locc))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_d23f8c26(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    wh = unifint(diff_lb, diff_ub, (1, 14))
    w = 2 * wh + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    numn = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    remcols = sample(remcols, numcols)
    inds = totuple(asindices(gi))
    locs = sample(inds, numn)
    for ij in locs:
        col = choice(remcols)
        gi = fill(gi, col, {ij})
        a, b = ij
        if b == w // 2:
            go = fill(go, col, {ij})
    return {'input': gi, 'output': go}


def generate_ce9e57f2(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nbars = unifint(diff_lb, diff_ub, (2, (w - 2) // 2))
    locopts = interval(1, w - 1, 1)
    barlocs = []
    for k in range(nbars):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        barlocs.append(loc)
        locopts = remove(loc, locopts)
        locopts = remove(loc + 1, locopts)
        locopts = remove(loc - 1, locopts)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    colss = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for j in barlocs:
        barloci = unifint(diff_lb, diff_ub, (1, h - 2))
        fullbar = connect((0, j), (barloci, j))
        halfbar = connect((0, j), (barloci // 2 if barloci % 2 == 1 else (barloci - 1) // 2, j))
        barcol = choice(colss)
        gi = fill(gi, barcol, fullbar)
        go = fill(go, barcol, fullbar)
        go = fill(go, 8, halfbar)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_b9b7f026(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    outcol = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 7)
        ow = randint(3, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            remcols = remove(col, remcols)
            gi = fill(gi, col, bd)
            succ += 1
            indss = indss - bd
            if outcol is None:
                outcol = col
                cands = totuple(backdrop(inbox(bd)))
                bd2 = backdrop(
                    frozenset(sample(cands, 2)) if len(cands) > 2 else frozenset(cands)
                )
                gi = fill(gi, bgc, bd2)
        tr += 1
    go = canvas(outcol, (1, 1))
    return {'input': gi, 'output': go}


def generate_6d75e8bb(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    nc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    fgc = choice(remcols)
    c = canvas(bgc, (h, w))
    bounds = asindices(c)
    ch = choice(totuple(bounds))
    shp = {ch}
    bounds = remove(ch, bounds)
    for j in range(nc - 1):
        shp.add(choice(totuple((bounds - shp) & mapply(neighbors, shp))))
    shp = normalize(shp)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    loc = (loci, locj)
    plcd = shift(shp, loc)
    gi = fill(c, fgc, plcd)
    go = fill(c, 2, backdrop(plcd))
    go = fill(go, fgc, plcd)
    return {'input': gi, 'output': go}


def generate_3f7978a0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, noisec, linec = sample(cols, 3)
    c = canvas(bgc, (h, w))
    oh = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * h))))
    oh = min(oh, h)
    ow = unifint(diff_lb, diff_ub, (4, max(4, int((2/3) * w))))
    ow = min(ow, w)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    nnoise = unifint(diff_lb, diff_ub, (0, (h * w) // 4))
    inds = totuple(asindices(c))
    noise = sample(inds, nnoise)
    gi = fill(c, noisec, noise)
    ulc = (loci, locj)
    lrc = (loci + oh - 1, locj + ow - 1)
    llc = (loci + oh - 1, locj)
    urc = (loci, locj + ow - 1)
    gi = fill(gi, linec, connect(ulc, llc))
    gi = fill(gi, linec, connect(urc, lrc))
    crns = {ulc, lrc, llc, urc}
    gi = fill(gi, noisec, crns)
    go = subgrid(crns, gi)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_e76a88a6(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, 5))
    objw = unifint(diff_lb, diff_ub, (2, 5))
    bounds = asindices(canvas(0, (objh, objw)))
    shp = {choice(totuple(bounds))}
    nc = unifint(diff_lb, diff_ub, (2, len(bounds) - 2))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(dneighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    dmyc = choice(remcols)
    remcols = remove(dmyc, remcols)
    oh, ow = shape(shp)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    shpp = shift(shp, (loci, locj))
    numco = unifint(diff_lb, diff_ub, (2, 8))
    colll = sample(remcols, numco)
    shppc = frozenset({(choice(colll), ij) for ij in shpp})
    while numcolors(shppc) == 1:
        shppc = frozenset({(choice(colll), ij) for ij in shpp})
    shppcn = normalize(shppc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shppc)
    go = tuple(e for e in gi)
    ub = ((h * w) / (oh * ow)) // 2
    ub = max(1, ub)
    numlocs = unifint(diff_lb, diff_ub, (1, ub))
    cnt = 0
    fails = 0
    maxfails = 5 * numlocs
    idns = (asindices(gi) - shpp) - mapply(dneighbors, shpp)
    idns = sfilter(idns, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
    while cnt < numlocs and fails < maxfails:
        if len(idns) == 0:
            break
        loc = choice(totuple(idns))
        plcd = shift(shppcn, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(idns):
            go = paint(go, plcd)
            gi = fill(gi, dmyc, plcdi)
            cnt += 1
            idns = (idns - plcdi) - mapply(dneighbors, plcdi)
        else:
            fails += 1
    return {'input': gi, 'output': go}


def generate_a61f2674(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, remove(1, interval(0, 10, 1)))
    w = unifint(diff_lb, diff_ub, (5, 28))
    h = unifint(diff_lb, diff_ub, (w // 2 + 1, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nbars = unifint(diff_lb, diff_ub, (2, w // 2))
    barlocs = []
    options = interval(0, w, 1)
    while len(options) > 0 and len(barlocs) < nbars:
        loc = choice(options)
        barlocs.append(loc)
        options = remove(loc, options)
        options = remove(loc + 1, options)
        options = remove(loc - 1, options)
    barheights = sample(interval(0, h, 1), nbars)
    for j, bh in zip(barlocs, barheights):
        gi = fill(gi, fgc, connect((0, j), (bh, j)))
        if bh == max(barheights):
            go = fill(go, 1, connect((0, j), (bh, j)))
        if bh == min(barheights):
            go = fill(go, 2, connect((0, j), (bh, j)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_ce4f8723(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    barcol = choice(remcols)
    remcols = remove(barcol, remcols)
    cola = choice(remcols)
    colb = choice(remove(cola, remcols))
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    numa = mp + deva
    numb = mp + devb
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gi = hconcat(hconcat(gia, gbar), gib)
    go = fill(canv, 3, set(a) | set(b))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_caa06a1f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    vp = unifint(diff_lb, diff_ub, (2, h//2-1))
    hp = unifint(diff_lb, diff_ub, (2, w//2-1))
    bgc = choice(cols)
    numc = unifint(diff_lb, diff_ub, (2, min(8, max(2, hp * vp))))
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numc)
    remcols = difference(remcols, ccols)
    tric = choice(remcols)
    obj = {(choice(ccols), ij) for ij in asindices(canvas(-1, (vp, hp)))}
    go = canvas(bgc, (h, w))
    gi = canvas(bgc, (h, w))
    for a in range(-vp, h+1, vp):
        for b in range(-hp, w+1, hp):
            go = paint(go, shift(obj, (a, b + 1)))
    for a in range(-vp, h+1, vp):
        for b in range(-hp, w+1, hp):
            gi = paint(gi, shift(obj, (a, b)))
    ioffs = unifint(diff_lb, diff_ub, (1, h - 2 * vp))
    joffs = unifint(diff_lb, diff_ub, (1, w - 2 * hp))
    for a in range(ioffs):
        gi = fill(gi, tric, connect((a, 0), (a, w - 1)))
    for b in range(joffs):
        gi = fill(gi, tric, connect((0, b), (h - 1, b)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_94f9d214(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 14))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    acol = choice(remcols)
    remcols = remove(acol, remcols)
    bcol = choice(remcols)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    numadev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numbdev = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numa = choice((numadev, h * w - numadev))
    numb = choice((numadev, h * w - numbdev))
    numa = min(max(1, numa), h * w - 1)
    numb = min(max(1, numb), h * w - 1)
    aset = sample(inds, numa)
    bset = sample(inds, numb)
    A = fill(c, acol, aset)
    B = fill(c, bcol, bset)
    gi = hconcat(A, B)
    res = (set(inds) - set(aset)) & (set(inds) - set(bset))
    go = fill(c, 2, res)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_feca6190(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 6))
    bgc = 0
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, min(w, 5)))
    ccols = sample(remcols, ncols)
    cands = interval(0, w, 1)
    locs = sample(cands, ncols)
    gi = canvas(bgc, (1, w))
    go = canvas(bgc, (w*ncols, w*ncols))
    for col, j in zip(ccols, locs):
        gi = fill(gi, col, {(0, j)})
        go = fill(go, col, shoot((w*ncols-1, j), UP_RIGHT))
    return {'input': gi, 'output': go}


def generate_d5d6de2d(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 16))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(1, 7)
        ow = randint(1, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            gi = fill(gi, col, box(bd))
            if oh > 2 and ow > 2:
                go = fill(go, 3, backdrop(inbox(bd)))
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_4612dd53(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ih = unifint(diff_lb, diff_ub, (5, h-1))
    iw = unifint(diff_lb, diff_ub, (5, w-1))
    bgc, col = sample(cols, 2)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bx = box(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    if choice((True, False)):
        locc = randint(loci + 2, loci + ih - 3)
        br = connect((locc, locj+1), (locc, locj + iw - 2))
    else:
        locc = randint(locj + 2, locj + iw - 3)
        br = connect((loci+1, locc), (loci + ih - 2, locc))
    c = canvas(bgc, (h, w))
    crns = sample(totuple(corners(bx)), 3)
    onbx = totuple(crns)
    rembx = difference(bx, crns)
    onbr = sample(totuple(br), 2)
    rembr = difference(br, onbr)
    noccbx = unifint(diff_lb, diff_ub, (0, len(rembx)))
    noccbr = unifint(diff_lb, diff_ub, (0, len(rembr)))
    occbx = sample(totuple(rembx), noccbx)
    occbr = sample(totuple(rembr), noccbr)
    c = fill(c, col, bx)
    c = fill(c, col, br)
    gi = fill(c, bgc, occbx)
    gi = fill(gi, bgc, occbr)
    go = fill(c, 2, occbx)
    go = fill(go, 2, occbr)
    if choice((True, False)):
        gi = fill(gi, bgc, br)
        go = fill(go, bgc, br)
    return {'input': gi, 'output': go}


def generate_1f642eb9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ih = unifint(diff_lb, diff_ub, (2, min(h - 4, 2 * (h // 3))))
    iw = unifint(diff_lb, diff_ub, (2, min(w - 4, 2 * (w // 3))))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    bgc, sqc = sample(cols, 2)
    remcols = difference(cols, (bgc, sqc))
    numcells = unifint(diff_lb, diff_ub, (1, 2 * ih + 2 * iw - 4))
    outs = []
    ins = []
    c1 = choice((True, False))
    c2 = choice((True, False))
    c3 = choice((True, False))
    c4 = choice((True, False))
    for a in range(loci + (not c1), loci + ih - (not c2)):
        outs.append((a, 0))
        ins.append((a, locj))
    for a in range(loci + (not c3), loci + ih - (not c4)):
        outs.append((a, w - 1))
        ins.append((a, locj + iw - 1))
    for b in range(locj + c1, locj + iw - (c3)):
        outs.append((0, b))
        ins.append((loci, b))
    for b in range(locj + (c2), locj + iw - (c4)):
        outs.append((h - 1, b))
        ins.append((loci + ih - 1, b))
    inds = interval(0, 2 * ih + 2 * iw - 4, 1)
    locs = sample(inds, numcells)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    outs = [e for j, e in enumerate(outs) if j in locs]
    ins = [e for j, e in enumerate(ins) if j in locs]
    c = canvas(bgc, (h, w))
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    gi = fill(c, sqc, bd)
    seq = [choice(ccols) for k in range(numcells)]
    for c, loc in zip(seq, outs):
        gi = fill(gi, c, {loc})
    go = tuple(e for e in gi)
    for c, loc in zip(seq, ins):
        go = fill(go, c, {loc})
    return {'input': gi, 'output': go}


def generate_681b3aeb(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    fullsuc = False
    while not fullsuc:
        hi = unifint(diff_lb, diff_ub, (2, 8))
        wi = unifint(diff_lb, diff_ub, (2, 8))
        h = unifint(diff_lb, diff_ub, ((3*hi, 30)))
        w = unifint(diff_lb, diff_ub, ((3*wi, 30)))
        c = canvas(-1, (hi, hi))
        bgc, ca, cb = sample(cols, 3)
        gi = canvas(bgc, (h, w))
        conda, condb = True, True
        while conda and condb:
            inds = totuple(asindices(c))
            pa = choice(inds)
            reminds = remove(pa, inds)
            pb = choice(reminds)
            reminds = remove(pb, reminds)
            A = {pa}
            B = {pb}
            for k in range(len(reminds)):
                acands = set(reminds) & mapply(dneighbors, A)
                bcands = set(reminds) & mapply(dneighbors, B)
                opts = []
                if len(acands) > 0:
                    opts.append(0)
                if len(bcands) > 0:
                    opts.append(1)
                idx = choice(opts)
                if idx == 0:
                    loc = choice(totuple(acands))
                    A.add(loc)
                else:
                    loc = choice(totuple(bcands))
                    B.add(loc)
                reminds = remove(loc, reminds)
            conda = len(A) == height(A) * width(A)
            condb = len(B) == height(B) * width(B)
        go = fill(c, ca, A)
        go = fill(go, cb, B)
        fullocs = totuple(asindices(gi))
        A = normalize(A)
        B = normalize(B)
        ha, wa = shape(A)
        hb, wb = shape(B)
        minisuc = False
        if not (ha > h or wa > w):
            for kkk in range(10):
                locai = randint(0, h - ha)
                locaj = randint(0, w - wa)
                plcda = shift(A, (locaj, locaj))
                remlocs = difference(fullocs, plcda)
                remlocs2 = sfilter(remlocs, lambda ij: ij[0] <= h - hb and ij[1] <= w - wb)
                if len(remlocs2) == 0:
                    continue
                ch = choice(remlocs2)
                plcdb = shift(B, (ch))
                if set(plcdb).issubset(set(remlocs2)):
                    minisuc = True
                    break
        if minisuc:
            fullsuc = True
    gi = fill(gi, ca, plcda)
    gi = fill(gi, cb, plcdb)
    return {'input': gi, 'output': go}


def generate_d364b489(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 6, 7, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 5))
    res = set()
    for j in range(num):
        if len(inds) == 0:
            break
        r = choice(inds)
        inds = remove(r, inds)
        inds = difference(inds, neighbors(r))
        inds = difference(inds, totuple(shift(apply(rbind(multiply, TWO), dneighbors(ORIGIN)), r)))
        res.add(r)
    gi = fill(gi, fgc, res)
    go = fill(gi, 7, shift(res, LEFT))
    go = fill(go, 6, shift(res, RIGHT))
    go = fill(go, 8, shift(res, DOWN))
    go = fill(go, 2, shift(res, UP))
    return {'input': gi, 'output': go}


def generate_25d8a9c8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = []
    go = []
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    ccols = sample(cols, ncols)
    for k in range(h):
        singlecol = choice((True, False))
        col = choice(ccols)
        row = repeat(col, w)
        if singlecol:
            gi.append(row)
            go.append(repeat(5, w))
        else:
            remcols = remove(col, ccols)
            nothercinv = unifint(diff_lb, diff_ub, (1, w - 1))
            notherc = w - 1 - nothercinv
            notherc = min(max(1, notherc), w - 1)
            row = list(row)
            indss = interval(0, w, 1)
            for j in sample(indss, notherc):
                row[j] = choice(remcols)
            gi.append(tuple(row))
            go.append(repeat(0, w))
    gi = tuple(gi)
    go = tuple(go)
    return {'input': gi, 'output': go}


def generate_bda2d7a6(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 14))
    ncols = unifint(diff_lb, diff_ub, (2, 10))
    cols = sample(colopts, ncols)
    colord = [choice(cols) for j in range(min(h, w))]
    shp = (h*2, w*2)
    gi = canvas(0, shp)
    for idx, (ci, co) in enumerate(zip(colord, colord[-1:] + colord[:-1])):
        ulc = (idx, idx)
        lrc = (h*2 - 1 - idx, w*2 - 1 - idx)
        bx = box(frozenset({ulc, lrc}))
        gi = fill(gi, ci, bx)
    I = gi
    objso = order(objects(I, T, F, F), compose(maximum, shape))
    if color(objso[0]) == color(objso[-1]):
        objso = (combine(objso[0], objso[-1]),) + objso[1:-1]
    res = mpapply(recolor, apply(color, objso), (objso[-1],) + objso[:-1])
    go = paint(gi, res)
    return {'input': gi, 'output': go}


def generate_a5f85a15(diff_lb: float, diff_ub: float) -> dict:
    colopts = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    startlocs = apply(toivec, interval(h - 1, 0, -1)) + apply(tojvec, interval(0, w, 1))
    cands = interval(0, h + w - 1, 1)
    num = unifint(diff_lb, diff_ub, (1, (h + w - 1) // 3))
    locs = []
    for k in range(num):
        if len(cands) == 0:
            break
        loc = choice(cands)
        locs.append(loc)
        cands = remove(loc, cands)
        cands = remove(loc - 1, cands)
        cands = remove(loc + 1, cands)
    locs = set([startlocs[loc] for loc in locs])
    bgc, fgc = sample(colopts, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for loc in locs:
        ln = order(shoot(loc, (1, 1)), first)
        gi = fill(gi, fgc, ln)
        go = fill(go, fgc, ln)
        go = fill(go, 4, ln[1::2])
    return {'input': gi, 'output': go}


def generate_32597951(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (2, h // 2))
    iw = unifint(diff_lb, diff_ub, (2, w // 2))
    bgc, noisec, fgc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    ndev = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    num = choice((ndev, h * w - ndev))
    num = min(max(num, 0), h * w)
    ofc = sample(inds, num)
    c = fill(c, noisec, ofc)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    tofillfc = bd & ofcolor(c, bgc)
    gi = fill(c, fgc, tofillfc)
    if len(tofillfc) > 0:
        go = fill(gi, 3, backdrop(tofillfc) & ofcolor(gi, noisec))
    else:
        go = gi
    return {'input': gi, 'output': go}


def generate_cf98881b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 9))
    bgc, barcol, cola, colb, colc = sample(cols, 5)
    canv = canvas(bgc, (h, w))
    inds = totuple(asindices(canv))
    gbar = canvas(barcol, (h, 1))
    mp = (h * w) // 2
    devrng = (0, mp)
    deva = unifint(diff_lb, diff_ub, devrng)
    devb = unifint(diff_lb, diff_ub, devrng)
    devc = unifint(diff_lb, diff_ub, devrng)
    sgna = choice((+1, -1))
    sgnb = choice((+1, -1))
    sgnc = choice((+1, -1))
    deva = sgna * deva
    devb = sgnb * devb
    devc = sgnc * devc
    numa = mp + deva
    numb = mp + devb
    numc = mp + devc
    numa = max(min(h * w - 1, numa), 1)
    numb = max(min(h * w - 1, numb), 1)
    numc = max(min(h * w - 1, numc), 1)
    a = sample(inds, numa)
    b = sample(inds, numb)
    c = sample(inds, numc)
    gia = fill(canv, cola, a)
    gib = fill(canv, colb, b)
    gic = fill(canv, colc, c)
    gi = hconcat(hconcat(hconcat(gia, gbar), hconcat(gib, gbar)), gic)
    go = fill(gic, colb, b)
    go = fill(go, cola, a)
    return {'input': gi, 'output': go}


def generate_41e4d17e(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(6, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (4, 4)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, bxs)
            go = fill(go, fgc, bxs)
            cen = center(bxs)
            frns = hfrontier(cen) | vfrontier(cen)
            kep = frns & ofcolor(go, bgc)
            go = fill(go, 6, kep)
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {'input': gi, 'output': go}


def generate_91714a58(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, targc = sample(cols, 2)
    remcols = remove(bgc, cols)
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    gi = canvas(bgc, (h, w))
    inds = totuple(asindices(gi))
    noise = sample(inds, nnoise)
    ih = randint(2, h // 2)
    iw = randint(2, w // 2)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    loc = (loci, locj)
    bd = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    go = fill(gi, targc, bd)
    for ij in noise:
        col = choice(remcols)
        gi = fill(gi, col, {ij})
    gi = fill(gi, targc, bd)
    return {'input': gi, 'output': go}


def generate_b60334d2(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 9))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (2, 2)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            cen = center(bxs)
            gi = fill(gi, fgc, {cen})
            go = fill(go, fgc, ineighbors(cen))
            go = fill(go, 1, dneighbors(cen))
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {'input': gi, 'output': go}


def generate_952a094c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ih = unifint(diff_lb, diff_ub, (4, h - 2))
    iw = unifint(diff_lb, diff_ub, (4, w - 2))
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    sp = (loci, locj)
    ep = (loci + ih - 1, locj + iw - 1)
    bx = box(frozenset({sp, ep}))
    bgc, fgc, a, b, c, d = sample(cols, 6)
    canv = canvas(bgc, (h, w))
    canvv = fill(canv, fgc, bx)
    gi = tuple(e for e in canvv)
    go = tuple(e for e in canvv)
    gi = fill(gi, a, {(loci + 1, locj + 1)})
    go = fill(go, a, {(loci + ih, locj + iw)})
    gi = fill(gi, b, {(loci + 1, locj + iw - 2)})
    go = fill(go, b, {(loci + ih, locj - 1)})
    gi = fill(gi, c, {(loci + ih - 2, locj + 1)})
    go = fill(go, c, {(loci - 1, locj + iw)})
    gi = fill(gi, d, {(loci + ih - 2, locj + iw - 2)})
    go = fill(go, d, {(loci - 1, locj - 1)})
    return {'input': gi, 'output': go}


def generate_b8cdaf2b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc, linc, dotc = sample(cols, 3)
    lin = connect((0, 0), (0, w - 1))
    winv = unifint(diff_lb, diff_ub, (2, w - 1))
    w2 = w - winv
    w2 = min(max(w2, 1), w - 2)
    locj = randint(1, w - w2 - 1)
    bar2 = connect((0, locj), (0, locj + w2 - 1))
    c = canvas(bgc, (h, w))
    gi = fill(c, linc, lin)
    gi = fill(gi, dotc, bar2)
    gi = fill(gi, linc, shift(bar2, (1, 0)))
    go = fill(gi, dotc, shoot((2, locj - 1), (1, -1)))
    go = fill(go, dotc, shoot((2, locj + w2), (1, 1)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_b548a754(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    hi = unifint(diff_lb, diff_ub, (4, h - 1))
    wi = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - hi)
    locj = randint(0, w - wi)
    bx = box(frozenset({(loci, locj), (loci + hi - 1, locj + wi - 1)}))
    ins = backdrop(inbox(bx))
    bgc, boxc, inc, dotc = sample(cols, 4)
    c = canvas(bgc, (h, w))
    go = fill(c, boxc, bx)
    go = fill(go, inc, ins)
    cutoff = randint(loci + 2, loci + hi - 2)
    bx2 = box(frozenset({(loci, locj), (cutoff, locj + wi - 1)}))
    ins2 = backdrop(inbox(bx2))
    gi = fill(c, boxc, bx2)
    gi = fill(gi, inc, ins2)
    locc = choice(totuple(connect((loci+hi-1, locj), (loci+hi-1, locj+wi-1))))
    gi = fill(gi, dotc, {locc})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_95990924(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    bx = box(frozenset({(0, 0), (3, 3)}))
    bd = backdrop(bx)
    maxtrials = 4 * num
    succ = 0
    tr = 0
    while succ < num and tr < maxtrials:
        loc = choice(totuple(inds))
        bxs = shift(bx, loc)
        if bxs.issubset(set(inds)):
            gi = fill(gi, fgc, inbox(bxs))
            go = fill(go, fgc, inbox(bxs))
            go = fill(go, 1, {loc})
            go = fill(go, 2, {add(loc, (0, 3))})
            go = fill(go, 3, {add(loc, (3, 0))})
            go = fill(go, 4, {add(loc, (3, 3))})
            inds = difference(inds, shift(bd, loc))
            succ += 1
        tr += 1
    return {'input': gi, 'output': go}


def generate_f1cefba8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    ih = unifint(diff_lb, diff_ub, (6, h - 1))
    iw = unifint(diff_lb, diff_ub, (6, w - 1))
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    bgc, ringc, inc = sample(cols, 3)
    obj = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    ring1 = box(obj)
    ring2 = inbox(obj)
    bd = backdrop(obj)
    c = canvas(bgc, (h, w))
    c = fill(c, inc, bd)
    c = fill(c, ringc, ring1 | ring2)
    cands = totuple(ring2 - corners(ring2))
    numc = unifint(diff_lb, diff_ub, (1, len(cands) // 2))
    locs = sample(cands, numc)
    gi = fill(c, inc, locs)
    lm = lowermost(ring2)
    hori = sfilter(locs, lambda ij: ij[0] > loci + 1 and ij[0] < lm)
    verti = difference(locs, hori)
    hlines = mapply(hfrontier, hori)
    vlines = mapply(vfrontier, verti)
    fulllocs = set(hlines) | set(vlines)
    topaintinc = fulllocs & ofcolor(c, bgc)
    topaintringc = fulllocs & ofcolor(c, inc)
    go = fill(c, inc, topaintinc)
    go = fill(go, ringc, topaintringc)
    return {'input': gi, 'output': go}


def generate_c444b776(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 9))
    w = unifint(diff_lb, diff_ub, (2, 9))
    nh = unifint(diff_lb, diff_ub, (1, 3))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgclinc = sample(cols, 2)
    bgc, linc = bgclinc
    remcols = difference(cols, bgclinc)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    numcol = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcol)
    numcels = unifint(diff_lb, diff_ub, (1, (h * w) // 2))
    cels = sample(inds, numcels)
    obj = {(choice(ccols), ij) for ij in cels}
    smallcpainted = paint(smallc, obj)
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    obj = asobject(smallcpainted)
    gi = paint(c, shift(obj, srcloc))
    remlocs = remove(srcloc, llocs)
    bobj = asobject(smallc)
    for rl in remlocs:
        gi = paint(gi, shift(bobj, rl))
    go = tuple(e for e in gi)
    for rl in remlocs:
        go = paint(go, shift(obj, rl))
    return {'input': gi, 'output': go}


def generate_97999447(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    opts = interval(0, h, 1)
    num = unifint(diff_lb, diff_ub, (1, h))
    locs = sample(opts, num)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in locs:
        col = choice(ccols)
        j = randint(0, w - 1)
        dot = (idx, j)
        gi = fill(gi, col, {dot})
        go = fill(go, col, {(idx, x) for x in range(j, w, 2)})
        go = fill(go, 5, {(idx, x) for x in range(j+1, w, 2)})
    return {'input': gi, 'output': go}


def generate_d89b689b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, sqc, a, b, c, d = sample(cols, 6)
    loci = randint(1, h - 3)
    locj = randint(1, w - 3)
    canv = canvas(bgc, (h, w))
    go = fill(canv, a, {(loci, locj)})
    go = fill(go, b, {(loci, locj+1)})
    go = fill(go, c, {(loci+1, locj)})
    go = fill(go, d, {(loci+1, locj+1)})
    inds = totuple(asindices(canv))
    aopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] < locj)
    bopts = sfilter(inds, lambda ij: ij[0] < loci and ij[1] > locj + 1)
    copts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] < locj)
    dopts = sfilter(inds, lambda ij: ij[0] > loci + 1 and ij[1] > locj + 1)
    aopts = order(aopts, lambda ij: manhattan({ij}, {(loci, locj)}))
    bopts = order(bopts, lambda ij: manhattan({ij}, {(loci, locj + 1)}))
    copts = order(copts, lambda ij: manhattan({ij}, {(loci + 1, locj)}))
    dopts = order(dopts, lambda ij: manhattan({ij}, {(loci + 1, locj + 1)}))
    aidx = unifint(diff_lb, diff_ub, (0, len(aopts) - 1))
    bidx = unifint(diff_lb, diff_ub, (0, len(bopts) - 1))
    cidx = unifint(diff_lb, diff_ub, (0, len(copts) - 1))
    didx = unifint(diff_lb, diff_ub, (0, len(dopts) - 1))
    loca = aopts[aidx]
    locb = bopts[bidx]
    locc = copts[cidx]
    locd = dopts[didx]
    gi = fill(canv, sqc, backdrop({(loci, locj), (loci + 1, locj + 1)}))
    gi = fill(gi, a, {loca})
    gi = fill(gi, b, {locb})
    gi = fill(gi, c, {locc})
    gi = fill(gi, d, {locd})
    return {'input': gi, 'output': go}


def generate_543a7ed5(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(4, 8)
        ow = randint(4, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            bdibd = backdrop(frozenset({(loci+1, locj+1), (loci + oh - 2, locj + ow - 2)}))
            go = fill(go, col, bdibd)
            go = fill(go, 3, box(bd))
            gi = fill(gi, col, bdibd)
            if oh > 5 and ow > 5 and randint(1, 10) != 1:
                ulci, ulcj = ulcorner(bdibd)
                lrci, lrcj = lrcorner(bdibd)
                aa = randint(ulci + 1, lrci - 1)
                aa = randint(ulci + 1, aa)
                bb = randint(ulcj + 1, lrcj - 1)
                bb = randint(ulcj + 1, bb)
                cc = randint(aa, lrci - 1)
                dd = randint(bb, lrcj - 1)
                cc = randint(cc, lrci - 1)
                dd = randint(dd, lrcj - 1)
                ins = backdrop({(aa, bb), (cc, dd)})
                go = fill(go, 4, ins)
                gi = fill(gi, bgc, ins)
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_a2fd1cf0(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 3, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 3))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi+1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj+1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 3))
    go = fill(gi, 8, connect((a, b), (a, d)))
    go = fill(go, 8, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 3, {(c, d)})
    return {'input': gi, 'output': go}


def generate_cdecee7f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    numc = unifint(diff_lb, diff_ub, (1, min(9, w)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    inds = interval(0, w, 1)
    locs = sample(inds, numc)
    locs = order(locs, identity)
    gi = canvas(bgc, (h, w))
    go = []
    for j in locs:
        iloc = randint(0, h - 1)
        col = choice(ccols)
        gi = fill(gi, col, {(iloc, j)})
        go.append(col)
    go = go + [bgc] * (9 - len(go))
    go = tuple(go)
    go = tuple([go[:3], go[3:6][::-1], go[6:]])
    return {'input': gi, 'output': go}


def generate_0962bcdd(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (2, 7))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    oh, ow = 5, 5
    subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + 4, locj + 4)})
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            ca, cb = sample(ccols, 2)
            cp = (loci + 2, locj + 2)
            lins1 = connect((loci, locj), (loci + 4, locj + 4))
            lins2 = connect((loci + 4, locj), (loci, locj + 4))
            lins12 = lins1 | lins2
            lins3 = connect((loci + 2, locj), (loci + 2, locj + 4))
            lins4 = connect((loci, locj + 2), (loci + 4, locj + 2))
            lins34 = lins3 | lins4
            go = fill(go, cb, lins34)
            go = fill(go, ca, lins12)
            gi = fill(gi, ca, {cp})
            gi = fill(gi, cb, dneighbors(cp))
            succ += 1
            indss = indss - bd
        tr += 1
    return {'input': gi, 'output': go}


def generate_dc0a314f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(diff_lb, diff_ub, (1, 2*h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2*h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2*h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 3, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_29623171(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 6))
    w = unifint(diff_lb, diff_ub, (2, 6))
    nh = unifint(diff_lb, diff_ub, (2, 4))
    nw = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    inds = totuple(asindices(smallc))
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    nmostc = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    mostc = sample(inds, nmostc)
    srcg = fill(smallc, fgc, mostc)
    obj = asobject(srcg)
    shftd = shift(obj, srcloc)
    gi = paint(c, shftd)
    go = fill(c, fgc, shftd)
    remlocs = remove(srcloc, llocs)
    gg = asobject(fill(smallc, bgc, inds))
    for rl in remlocs:
        noth = unifint(diff_lb, diff_ub, (0, nmostc))
        otherg = fill(smallc, fgc, sample(inds, noth))
        gi = paint(gi, shift(asobject(otherg), rl))
        if noth == nmostc:
            go = fill(go, fgc, shift(obj, rl))
        else:
            go = paint(go, shift(gg, rl))
    return {'input': gi, 'output': go}


def generate_d4a91cb9(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 4, 8))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    gloci = unifint(diff_lb, diff_ub, (1, h - 1))
    glocj = unifint(diff_lb, diff_ub, (1, w - 1))
    gloc = (gloci, glocj)
    bgc = choice(cols)
    g = canvas(bgc, (h, w))
    g = fill(g, 8, {gloc})
    g = rot180(g)
    glocinv = center(ofcolor(g, 8))
    glocinvi, glocinvj = glocinv
    rloci = unifint(diff_lb, diff_ub, (glocinvi+1, h - 1))
    rlocj = unifint(diff_lb, diff_ub, (glocinvj+1, w - 1))
    rlocinv = (rloci, rlocj)
    g = fill(g, 2, {rlocinv})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(g)
    a, b = center(ofcolor(gi, 2))
    c, d = center(ofcolor(gi, 8))
    go = fill(gi, 4, connect((a, b), (a, d)))
    go = fill(go, 4, connect((a, d), (c, d)))
    go = fill(go, 2, {(a, b)})
    go = fill(go, 8, {(c, d)})
    return {'input': gi, 'output': go}


def generate_60b61512(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(7, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(2, 7)
        ow = randint(2, 7)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        indsss = asindices(canvas(-1, (oh, ow)))
        chch = choice(totuple(indsss))
        obj = {chch}
        indsss = remove(chch, indsss)
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = choice((numcd, oh * ow - numcd))
        numc = min(max(2, numc), oh * ow - 1)
        for k in range(numc):
            obj.add(choice(totuple(indsss & mapply(neighbors, obj))))
            indsss = indsss - obj
        oh, ow = shape(obj)
        obj = shift(obj, (loci, locj))
        bd = backdrop(obj)
        col = choice(ccols)
        if bd.issubset(indss):
            gi = fill(gi, col, obj)
            go = fill(go, 7, bd)
            go = fill(go, col, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_4938f0c2(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 31))
    w = unifint(diff_lb, diff_ub, (10, 31))
    oh = unifint(diff_lb, diff_ub, (2, (h - 3) // 2))
    ow = unifint(diff_lb, diff_ub, (2, (w - 3) // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    cc = choice(remcols)
    remcols = remove(cc, remcols)
    objc = choice(remcols)
    sg = canvas(bgc, (oh, ow))
    locc = (oh - 1, ow - 1)
    sg = fill(sg, cc, {locc})
    reminds = totuple(remove(locc, asindices(sg)))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
    cells = sample(reminds, ncells)
    while ncells == 4 and shape(cells) == (2, 2):
        ncells = unifint(diff_lb, diff_ub, (1, max(1, int((2/3) * oh * ow))))
        cells = sample(reminds, ncells)
    sg = fill(sg, objc, cells)
    G1 = sg
    G2 = vmirror(sg)
    G3 = hmirror(sg)
    G4 = vmirror(hmirror(sg))
    vbar = canvas(bgc, (oh, 1))
    hbar = canvas(bgc, (1, ow))
    cp = canvas(cc, (1, 1))
    topg = hconcat(hconcat(G1, vbar), G2)
    botg = hconcat(hconcat(G3, vbar), G4)
    ggm = hconcat(hconcat(hbar, cp), hbar)
    GG = vconcat(vconcat(topg, ggm), botg)
    gg = asobject(GG)
    canv = canvas(bgc, (h, w))
    loci = randint(0, h - 2 * oh - 1)
    locj = randint(0, w - 2 * ow - 1)
    loc = (loci, locj)
    go = paint(canv, shift(gg, loc))
    gi = paint(canv, shift(asobject(sg), loc))
    gi = fill(gi, cc, ofcolor(go, cc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    ccpi, ccpj = center(ofcolor(gi, cc))
    gi = gi[:ccpi] + gi[ccpi+1:]
    gi = tuple(r[:ccpj] + r[ccpj + 1:] for r in gi)
    go = go[:ccpi] + go[ccpi+1:]
    go = tuple(r[:ccpj] + r[ccpj + 1:] for r in go)
    return {'input': gi, 'output': go}


def generate_a8d7556c(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    fgc = choice(cols)
    c = canvas(fgc, (h, w))
    numblacks = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = totuple(asindices(c))
    blacks = sample(inds, numblacks)
    gi = fill(c, 0, blacks)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    sqlocs = sample(inds, numsq)
    gi = fill(gi, 0, shift(sqlocs, (0, 0)))
    gi = fill(gi, 0, shift(sqlocs, (0, 1)))
    gi = fill(gi, 0, shift(sqlocs, (1, 0)))
    gi = fill(gi, 0, shift(sqlocs, (1, 1)))
    go = tuple(e for e in gi)
    for a in range(h - 1):
        for b in range(w - 1):
            if gi[a][b] == 0 and gi[a+1][b] == 0 and gi[a][b+1] == 0 and gi[a+1][b+1] == 0:
                go = fill(go, 2, {(a, b), (a+1, b), (a, b+1), (a+1, b+1)})
    return {'input': gi, 'output': go}


def generate_007bbfb7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    c = canvas(0, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    inds = totuple(asindices(c))
    locs = sample(inds, numc)
    fgc = choice(cols)
    gi = fill(c, fgc, locs)
    go = canvas(0, (h**2, w**2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    return {'input': gi, 'output': go}


def generate_b190f7f5(diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(1, numc), h * w - 1)
    numcd2 = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc2 = choice((numcd2, h * w - numcd2))
    numc2 = min(max(2, numc2), h * w - 1)
    inds = totuple(asindices(c))
    srclocs = sample(inds, numc)
    srccol = choice(cols)
    remcols = remove(srccol, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    trglocs = sample(inds, numc2)
    ccols = sample(remcols, numcols)
    fixc1 = choice(ccols)
    trgobj = [(fixc1, trglocs[0]), (choice(remove(fixc1, ccols)), trglocs[1])] + [(choice(ccols), ij) for ij in trglocs[2:]]
    trgobj = frozenset(trgobj)
    gisrc = fill(c, srccol, srclocs)
    gitrg = paint(c, trgobj)
    catf = choice((hconcat, vconcat))
    ordd = choice(([gisrc, gitrg], [gitrg, gisrc]))
    gi = catf(*ordd)
    go = canvas(bgc, (h**2, w**2))
    for loc in trglocs:
        a, b = loc
        go = fill(go, gitrg[a][b], shift(srclocs, multiply(loc, (h, w))))
    return {'input': gi, 'output': go}


def generate_2bcee788(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 20))
    w = unifint(diff_lb, diff_ub, (2, 10))
    bgc, sepc, objc = sample(cols, 3)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    spi = randint(0, h - 1)
    sp = (spi, w - 1)
    shp = {sp}
    numcellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcellsd, h * w - numcellsd))
    numc = min(max(2, numc), h * w - 1)
    reminds = set(remove(sp, inds))
    for k in range(numc):
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    while width(shp) == 1:
        shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
    c2 = fill(c, objc, shp)
    borderinds = sfilter(shp, lambda ij: ij[1] == w - 1)
    c3 = fill(c, sepc, borderinds)
    gimini = asobject(hconcat(c2, vmirror(c3)))
    gomini = asobject(hconcat(c2, vmirror(c2)))
    fullh = unifint(diff_lb, diff_ub, (h+1, 30))
    fullw = unifint(diff_lb, diff_ub, (2*w+1, 30))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(0, fullh - h)
    locj = randint(0, fullw - 2 * w)
    loc = (loci, locj)
    gi = paint(fullg, gimini)
    go = paint(fullg, gomini)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    go = replace(go, bgc, 3)
    return {'input': gi, 'output': go}


def generate_a3df8b1e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 10))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    changing = True
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    return {'input': gi, 'output': go}


def generate_80af3007(diff_lb: float, diff_ub: float) -> dict:
    fullcols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    bgc = choice(fullcols)
    cols = remove(bgc, fullcols)
    c = canvas(bgc, (h, w))
    numcd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    numc = choice((numcd, h * w - numcd))
    numc = min(max(0, numc), h * w)
    inds = totuple(asindices(c))
    locs = tuple(set(sample(inds, numc)) | set(sample(totuple(corners(inds)), 3)))
    fgc = choice(cols)
    gi = fill(c, fgc, locs)
    go = canvas(bgc, (h**2, w**2))
    for loc in locs:
        go = fill(go, fgc, shift(locs, multiply(loc, (h, w))))
    fullh = unifint(diff_lb, diff_ub, (h**2+2, 30))
    fullw = unifint(diff_lb, diff_ub, (w**2+2, 30))
    fullg = canvas(bgc, (fullh, fullw))
    loci = randint(1, fullh - h**2 - 1)
    locj = randint(1, fullw - w**2 - 1)
    loc = (loci, locj)
    giups = hupscale(vupscale(gi, h), w)
    gi = paint(fullg, shift(asobject(giups), loc))
    return {'input': gi, 'output': go}


def generate_e50d258f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    padcol = choice(remcols)
    remcols = remove(padcol, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 10))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    bound = None
    go = None
    while succ < num and tr <= maxtrials:
        if len(remcols) == 0 or len(indss) == 0:
            break
        oh = randint(3, 8)
        ow = randint(3, 8)
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        if bd.issubset(indss):
            numcc = unifint(diff_lb, diff_ub, (1, 7))
            ccols = sample(remcols, numcc)
            if succ == 0:
                numred = unifint(diff_lb, diff_ub, (1, oh * ow))
                bound = numred
            else:
                numred = unifint(diff_lb, diff_ub, (0, min(oh * ow, bound - 1)))
            cc = canvas(choice(ccols), (oh, ow))
            cci = asindices(cc)
            subs = sample(totuple(cci), numred)
            obj1 = {(choice(ccols), ij) for ij in cci - set(subs)}
            obj2 = {(2, ij) for ij in subs}
            obj = obj1 | obj2
            gi = paint(gi, shift(obj, (loci, locj)))
            if go is None:
                go = paint(cc, obj)
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_0e206a2e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, acol, bcol, ccol, Dcol = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nsrcs = unifint(diff_lb, diff_ub, (1, min(h, w) // 5))
    srcs = []
    abclist = []
    maxtrforsrc = 5 * nsrcs
    trforsrc = 0
    srcsucc = 0
    while trforsrc < maxtrforsrc and srcsucc < nsrcs:
        trforsrc += 1
        objsize = unifint(diff_lb, diff_ub, (5, 20))
        bb = asindices(canvas(-1, (7, 7)))
        sp = choice(totuple(bb))
        bb = remove(sp, bb)
        shp = {sp}
        for k in range(objsize - 1):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while 1 in shape(shp):
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        while len(set([x - y for x, y in shp])) == 1 or len(set([x + y for x, y in shp])) == 1:
            shp.add(choice(totuple((bb - shp) & mapply(dneighbors, shp))))
        shp = normalize(shp)
        shp = list(shp)
        shuffle(shp)
        a, b, c = shp[:3]
        while 1 in shape({a, b, c}) or (len(set([x - y for x, y in {a, b, c}])) == 1 or len(set([x + y for x, y in {a, b, c}])) == 1):
            shuffle(shp)
            a, b, c = shp[:3]
        if sorted(shape({a, b, c})) in abclist:
            continue
        D = shp[3:]
        markers = {(acol, a), (bcol, b), (ccol, c)}
        obj = markers | {(Dcol, ij) for ij in D}
        obj = frozenset(obj)
        oh, ow = shape(obj)
        opts = sfilter(inds, lambda ij: shift(set(shp), ij).issubset(inds))
        if len(opts) == 0:
            continue
        loc = choice(totuple(opts))
        srcsucc += 1
        gi = paint(gi, shift(obj, loc))
        shpplcd = shift(set(shp), loc)
        go = fill(go, -1, shpplcd)
        inds = (inds - shpplcd) - mapply(neighbors, shpplcd)
        srcs.append((obj, markers))
        abclist.append(sorted(shape({a, b, c})))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    maxtrials = 10 * num
    tr = 0
    succ = 0
    while succ < num and tr < maxtrials:
        mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
        fn = choice(mfs)
        gi = fn(gi)
        go = fn(go)
        aigo = asindices(go)
        fullinds = ofcolor(go, bgc) - mapply(neighbors, aigo - ofcolor(go, bgc))
        obj, markers = choice(srcs)
        shp = toindices(obj)
        if len(fullinds) == 0:
            break
        loctr = choice(totuple(fullinds))
        xx = shift(shp, loctr)
        if xx.issubset(fullinds):
            succ += 1
            gi = paint(gi, shift(markers, loctr))
            go = paint(go, shift(obj, loctr))
        tr += 1
    go = replace(go, -1, bgc)
    return {'input': gi, 'output': go}


def generate_b230c067(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        oh = unifint(diff_lb, diff_ub, (2, h // 3 - 1))
        ow = unifint(diff_lb, diff_ub, (2, w // 3 - 1))
        numcd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        numc = choice((numcd, oh * ow - numcd))
        numca = min(max(2, numc), oh * ow - 2)
        bounds = asindices(canvas(-1, (oh, ow)))
        sp = choice(totuple(bounds))
        shp = {sp}
        for k in range(numca):
            ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
            shp.add(ij)
        shpa = normalize(shp)
        shpb = set(normalize(shp))
        mxnch = oh * ow - len(shpa)
        nchinv = unifint(diff_lb, diff_ub, (1, mxnch))
        nch = mxnch - nchinv
        nch = min(max(1, nch), mxnch)
        for k in range(nch):
            ij = choice(totuple((bounds - shpb) & mapply(neighbors, shpb)))
            shpb.add(ij)
        if choice((True, False)):
            shpa, shpb = shpb, shpa
        bgc, fgc = sample(cols, 2)
        c = canvas(bgc, (h, w))
        inds = asindices(c)
        acands = sfilter(inds, lambda ij: ij[0] <= h - height(shpa) and ij[1] <= w - width(shpa))
        aloc = choice(totuple(acands))
        aplcd = shift(shpa, aloc)
        gi = fill(c, fgc, aplcd)
        go = fill(c, 2, aplcd)
        maxtrials = 10
        tr = 0
        succ = 0
        inds = (inds - aplcd) - mapply(neighbors, aplcd)
        inds = sfilter(inds, lambda ij: ij[0] <= h - height(shpb) and ij[1] <= w - width(shpb))
        while succ < 2 and tr <= maxtrials:
            if len(inds) == 0:
                break
            loc = choice(totuple(inds))
            plcbd = shift(shpb, loc)
            if plcbd.issubset(inds):
                gi = fill(gi, fgc, plcbd)
                go = fill(go, 1, plcbd)
                succ += 1
                inds = (inds - plcbd) - mapply(neighbors, plcbd)
            tr += 1
        if succ == 2:
            break
    return {'input': gi, 'output': go}


def generate_db93a21d(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 3))
    h = unifint(diff_lb, diff_ub, (12, 31))
    w = unifint(diff_lb, diff_ub, (12, 32))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(1, h // 4)
        ow = oh
        fullh = 4 * oh
        fullw = 4 * ow
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - fullh and ij[1] < w - fullw))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        bigobj = backdrop(frozenset({(loci, locj), (loci + fullh - 1, locj + fullw - 1)}))
        smallobj = backdrop(frozenset({(loci+oh, locj+ow), (loci + fullh - 1 - oh, locj + fullw - 1 - ow)}))
        if bigobj.issubset(indss | ofcolor(go, 3)):
            gi = fill(gi, fgc, smallobj)
            go = fill(go, 3, bigobj)
            go = fill(go, fgc, smallobj)
            strp = mapply(rbind(shoot, (1, 0)), connect(lrcorner(smallobj), llcorner(smallobj)))
            go = fill(go, 1, ofcolor(go, bgc) & strp)
            succ += 1
            indss = indss - bigobj
        tr += 1
    gi = gi[1:]
    go = go[1:]
    gi = tuple(r[1:-1] for r in gi)
    go = tuple(r[1:-1] for r in go)
    return {'input': gi, 'output': go}


def generate_1e32b0e9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 6))
    w = unifint(diff_lb, diff_ub, (4, 6))
    nh = unifint(diff_lb, diff_ub, (1, 4))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 3))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h+1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    srcloc = choice(llocs)
    remlocs = remove(srcloc, llocs)
    ncells = unifint(diff_lb, diff_ub, (0, (h - 2) * (w - 2) - 1))
    smallc2 = canvas(bgc, (h-2, w - 2))
    inds = asindices(smallc2)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(ncells):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = shift(shp, (1, 1))
    gg = asobject(fill(smallc, fgc, shp))
    gg2 = asobject(fill(smallc, linc, shp))
    gi = paint(c, shift(gg, srcloc))
    go = tuple(e for e in gi)
    ncc = ncells + 1
    for rl in remlocs:
        nleft = randint(0, ncc)
        subobj = sample(totuple(shp), nleft)
        sg2 = asobject(fill(smallc, fgc, subobj))
        gi = paint(gi, shift(sg2, rl))
        go = paint(go, shift(gg2, rl))
        go = fill(go, fgc, shift(subobj, rl))
    return {'input': gi, 'output': go}


def generate_6773b310(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (2, 5))
    nw = unifint(diff_lb, diff_ub, (2, 5))
    bgc, linc, fgc = sample(cols, 3)
    fullh = h * nh + (nh - 1)
    fullw = w * nw + (nw - 1)
    c = canvas(linc, (fullh, fullw))
    smallc = canvas(bgc, (h, w))
    llocs = set()
    for a in range(0, fullh, h + 1):
        for b in range(0, fullw, w + 1):
            llocs.add((a, b))
    llocs = tuple(llocs)
    nbldev = unifint(diff_lb, diff_ub, (0, (nh * nw) // 2))
    nbl = choice((nbldev, nh * nw - nbldev))
    nbl = min(max(1, nbl), nh * nw - 1)
    bluelocs = sample(llocs, nbl)
    bglocs = difference(llocs, bluelocs)
    inds = totuple(asindices(smallc))
    gi = tuple(e for e in c)
    go = canvas(bgc, (nh, nw))
    for ij in bluelocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 2)))
        gi = paint(gi, shift(subg, ij))
        a, b = ij
        loci = a // (h+1)
        locj = b // (w+1)
        go = fill(go, 1, {(loci, locj)})
    for ij in bglocs:
        subg = asobject(fill(smallc, fgc, sample(inds, 1)))
        gi = paint(gi, shift(subg, ij))
    return {'input': gi, 'output': go}


def generate_6ecd11f4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 7))
    w = unifint(diff_lb, diff_ub, (2, 7))
    bgc, fgc = sample(cols, 2)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    inds = asindices(canvas(bgc, (h, w)))
    nlocsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nlocs = choice((nlocsd, h * w - nlocsd))
    nlocs = min(max(3, nlocs), h * w - 1)
    sp = choice(totuple(inds))
    inds = remove(sp, inds)
    shp = {sp}
    for j in range(nlocs):
        ij = choice(totuple((inds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    shp = normalize(shp)
    h, w = shape(shp)
    canv = canvas(bgc, (h, w))
    objbase = fill(canv, fgc, shp)
    maxhscf = (2*h+h+1) // h
    maxwscf = (2*w+w+1) // w
    hscf = unifint(diff_lb, diff_ub, (2, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (2, maxwscf))
    obj = asobject(hupscale(vupscale(objbase, hscf), wscf))
    oh, ow = shape(obj)
    inds = asindices(canv)
    objx = {(choice(ccols), ij) for ij in inds}
    if len(palette(objx)) == 1:
        objxodo = first(objx)
        objx = insert((choice(remove(objxodo[0], ccols)), objxodo[1]), remove(objxodo, objx))
    fullh = unifint(diff_lb, diff_ub, (hscf*h+h+1, 30))
    fullw = unifint(diff_lb, diff_ub, (wscf*w+w+1, 30))
    gi = canvas(bgc, (fullh, fullw))
    fullinds = asindices(gi)
    while True:
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        loc = (loci, locj)
        gix = paint(gi, shift(obj, loc))
        ofc = ofcolor(gix, fgc)
        delt = (fullinds - ofc)
        delt2 = delt - mapply(neighbors, ofc)
        scands = sfilter(
            delt2,
            lambda ij: ij[0] <= fullh - oh and ij[1] <= fullw - ow
        )
        if len(scands) == 0:
            continue
        locc = choice(totuple(scands))
        shftd = shift(objx, locc)
        if toindices(shftd).issubset(delt2):
            gi = paint(gix, shftd)
            break
    go = paint(canv, objx)
    go = fill(go, bgc, ofcolor(objbase, bgc))
    return {'input': gi, 'output': go}


def generate_8403a5d5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    loccinv = unifint(diff_lb, diff_ub, (1, w - 1))
    locc = w - loccinv
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    idx = (h - 1, locc)
    gi = fill(c, fgc, {idx})
    go = canvas(bgc, (h, w))
    for j in range(locc, w, 2):
        go = fill(go, fgc, connect((0, j), (h - 1, j)))
    for j in range(locc+1, w, 4):
        go = fill(go, 5, {(0, j)})
    for j in range(locc+3, w, 4):
        go = fill(go, 5, {(h-1, j)})
    return {'input': gi, 'output': go}


def generate_941d9a10(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    opts = interval(2, (h-1)//2 + 1, 2)
    nhidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nh = opts[nhidx]
    opts = interval(2, (w-1)//2 + 1, 2)
    nwidx = unifint(diff_lb, diff_ub, (0, len(opts) - 1))
    nw = opts[nwidx]
    bgc, fgc = sample(cols, 2)
    hgrid = canvas(bgc, (2*nh+1, w))
    for j in range(1, h, 2):
        hgrid = fill(hgrid, fgc, connect((j, 0), (j, w)))
    for k in range(h - (2*nh+1)):
        loc = randint(0, height(hgrid) - 1)
        hgrid = hgrid[:loc] + canvas(bgc, (1, w)) + hgrid[loc:]
    wgrid = canvas(bgc, (2*nw+1, h))
    for j in range(1, w, 2):
        wgrid = fill(wgrid, fgc, connect((j, 0), (j, h)))
    for k in range(w - (2*nw+1)):
        loc = randint(0, height(wgrid) - 1)
        wgrid = wgrid[:loc] + canvas(bgc, (1, h)) + wgrid[loc:]
    wgrid = dmirror(wgrid)
    gi = canvas(bgc, (h, w))
    fronts = ofcolor(hgrid, fgc) | ofcolor(wgrid, fgc)
    gi = fill(gi, fgc, fronts)
    objs = objects(gi, T, T, F)
    objs = colorfilter(objs, bgc)
    blue = argmin(objs, lambda o: leftmost(o) + uppermost(o))
    green = argmax(objs, lambda o: leftmost(o) + uppermost(o))
    f1 = lambda o: len(sfilter(objs, lambda o2: leftmost(o2) < leftmost(o))) == len(sfilter(objs, lambda o2: leftmost(o2) > leftmost(o)))
    f2 = lambda o: len(sfilter(objs, lambda o2: uppermost(o2) < uppermost(o))) == len(sfilter(objs, lambda o2: uppermost(o2) > uppermost(o)))
    red = extract(objs, lambda o: f1(o) and f2(o))
    go = fill(gi, 1, blue)
    go = fill(go, 3, green)
    go = fill(go, 2, red)
    return {'input': gi, 'output': go}


def generate_b0c4d837(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    oh = unifint(diff_lb, diff_ub, (3, h - 1))
    ow = unifint(diff_lb, diff_ub, (3, w - 1))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, boxc, fillc = sample(cols, 3)
    subg = canvas(boxc, (oh, ow))
    subg2 = canvas(fillc, (oh-1, ow-2))
    ntofill = unifint(diff_lb, diff_ub, (1, min(9, oh-2)))
    for j in range(ntofill):
        subg2 = fill(subg2, bgc, connect((j, 0), (j, ow-2)))
    subg = paint(subg, shift(asobject(subg2), (0, 1)))
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(asobject(subg), (loci, locj)))
    go = repeat(fillc, ntofill) + repeat(bgc, 9 - ntofill)
    go = (go[:3], go[3:6][::-1], go[6:])
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    return {'input': gi, 'output': go}


def generate_0a938d79(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 29))
    w = unifint(diff_lb, diff_ub, (h+1, 30))
    bgc, cola, colb = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    locja = unifint(diff_lb, diff_ub, (3, w - 2))
    locjb = unifint(diff_lb, diff_ub, (1, locja - 2))
    locia = choice((0, h-1))
    locib = choice((0, h-1))
    gi = fill(gi, cola, {(locia, locja)})
    gi = fill(gi, colb, {(locib, locjb)})
    ofs = -2 * (locja-locjb)
    for aa in range(locja, -1, ofs):
        go = fill(go, cola, connect((0, aa), (h-1, aa)))
    for bb in range(locjb, -1, ofs):    
        go = fill(go, colb, connect((0, bb), (h-1, bb)))
    rotf = choice((rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_b7249182(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    ih = unifint(diff_lb, diff_ub, (3, (h-1)//2))
    bgc, ca, cb = sample(cols, 3)
    subg = canvas(bgc, (ih, 5))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    subg = fill(subg, ca, connect((0, 2), (ih-2, 2)))
    subg = fill(subg, ca, connect((ih-2, 0), (ih-2, 4)))
    subg = fill(subg, ca, {(ih-1, 0)})
    subga = fill(subg, ca, {(ih-1, 4)})
    subgb = replace(subga, ca, cb)
    subg = vconcat(subga, hmirror(subgb))
    loci = randint(0, h-2*ih)
    locj = randint(0, w-5)
    obj = asobject(subg)
    obj = shift(obj, (loci, locj))
    gi = fill(gi, ca, {(loci, locj+2)})
    gi = fill(gi, cb, {(loci+2*ih-1, locj+2)})
    go = paint(go, obj)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_7b6016b9(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, remove(2, interval(0, 10, 1)))
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (5, 30))
        bgc, fgc = sample(cols, 2)
        numl = unifint(diff_lb, diff_ub, (4, min(h, w)))
        gi = canvas(bgc, (h, w))
        jint = interval(0, w, 1)
        iint = interval(0, h, 1)
        iopts = interval(1, h-1, 1)
        jopts = interval(1, w-1, 1)
        numlh = randint(numl//3, numl//3*2)
        numlw = numl - numlh
        for k in range(numlh):
            if len(iopts) == 0:
                continue
            loci = choice(iopts)
            iopts = remove(loci, iopts)
            iopts = remove(loci+1, iopts)
            iopts = remove(loci-1, iopts)
            a, b = sample(jint, 2)
            a = randint(0, a)
            b = randint(b, w - 1)
            gi = fill(gi, fgc, connect((loci, a), (loci, b)))
        for k in range(numlw):
            if len(jopts) == 0:
                continue
            locj = choice(jopts)
            jopts = remove(locj, jopts)
            jopts = remove(locj+1, jopts)
            jopts = remove(locj-1, jopts)
            a, b = sample(iint, 2)
            a = randint(0, a)
            b = randint(b, h - 1)
            gi = fill(gi, fgc, connect((a, locj), (b, locj)))
        objs = objects(gi, T, F, F)
        bgobjs = colorfilter(objs, bgc)
        tofill = toindices(mfilter(bgobjs, compose(flip, rbind(bordering, gi))))
        if len(tofill) > 0:
            break
    tofix = mapply(neighbors, tofill) - tofill
    gi = fill(gi, fgc, tofix)
    go = fill(gi, 2, tofill)
    go = replace(go, bgc, 3)
    return {'input': gi, 'output': go}


def generate_72ca375d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    srcobjh = unifint(diff_lb, diff_ub, (2, 8))
    srcobjwh = unifint(diff_lb, diff_ub, (1, 4))
    bnds = asindices(canvas(-1, (srcobjh, srcobjwh)))
    spi = randint(0, srcobjh - 1)
    sp = (spi, srcobjwh - 1)
    srcobj = {sp}
    bnds = remove(sp, bnds)
    ncellsd = unifint(diff_lb, diff_ub, (0, (srcobjh * srcobjwh) // 2))
    ncells1 = choice((ncellsd, srcobjh * srcobjwh - ncellsd))
    ncells2 = unifint(diff_lb, diff_ub, (1, srcobjh * srcobjwh))
    ncells = (ncells1 + ncells2) // 2
    ncells = min(max(1, ncells), srcobjh * srcobjwh, (h * w) // 2 - 1)
    for k in range(ncells - 1):
        srcobj.add(choice(totuple((bnds - srcobj) & mapply(neighbors, srcobj))))
    srcobj = normalize(srcobj)
    srcobj = srcobj | shift(vmirror(srcobj), (0, width(srcobj)))
    srcobjh, srcobjw = shape(srcobj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    trgc = choice(remcols)
    go = canvas(bgc, (srcobjh, srcobjw))
    go = fill(go, trgc, srcobj)
    loci = randint(0, h - srcobjh)
    locj = randint(0, w - srcobjw)
    locc = (loci, locj)
    gi = canvas(bgc, (h, w))
    shftd = shift(srcobj, locc)
    gi = fill(gi, trgc, shftd)
    indss = asindices(gi)
    indss = (indss - shftd) - mapply(neighbors, shftd)
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    remcands = asindices(canvas(-1, (8, 8))) - srcobj
    while succ < nobjs and tr <= maxtrials:
        if len(indss) == 0:
            break
        while True:
            newobj = {e for e in srcobj}
            numperti = unifint(diff_lb, diff_ub, (1, 63))
            numpert = 64 - numperti
            for np in range(numpert):
                isadd = choice((True, False))
                if isadd and len(newobj) < 64:
                    cndds = totuple((remcands - newobj) & mapply(neighbors, newobj))
                    if len(cndds) == 0:
                        break
                    newobj.add(choice(cndds))
                if not isadd and len(newobj) > 2:
                    newobj = remove(choice(totuple(newobj)), newobj)
            newobj = normalize(newobj)
            a, b = shape(newobj)
            cc = canvas(-1, (a+2, b+2))
            cc2 = compress(fill(cc, -2, shift(newobj, (1, 1))))
            newobj = toindices(argmax(colorfilter(objects(cc2, T, T, F), -2), size))
            if newobj != vmirror(newobj):
                break
        col = choice(remcols)
        loccands = sfilter(indss, lambda ij: shift(newobj, ij).issubset(indss))
        if len(loccands) == 0:
            tr += 1
            continue
        locc = choice(totuple(loccands))
        newobj = shift(newobj, locc)
        gi = fill(gi, col, newobj)
        succ += 1
        indss = (indss - newobj) - mapply(neighbors, newobj)
    return {'input': gi, 'output': go}


def generate_673ef223(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    barh = unifint(diff_lb, diff_ub, (2, (h-1)//2))
    ncells = unifint(diff_lb, diff_ub, (1, barh))
    bgc, barc, dotc = sample(cols, 3)
    sg = canvas(bgc, (barh, w))
    topsgi = fill(sg, barc, connect((0, 0), (barh-1, 0)))
    botsgi = vmirror(topsgi)
    topsgo = tuple(e for e in topsgi)
    botsgo = tuple(e for e in botsgi)
    iloccands = interval(0, barh, 1)
    ilocs = sample(iloccands, ncells)
    for k in ilocs:
        jloc = randint(2, w - 2)
        topsgi = fill(topsgi, dotc, {(k, jloc)})
        topsgo = fill(topsgo, 4, {(k, jloc)})
        topsgo = fill(topsgo, dotc, connect((k, 1), (k, jloc-1)))
        botsgo = fill(botsgo, dotc, connect((k, 0), (k, w - 2)))
    outpi = (topsgi, botsgi)
    outpo = (topsgo, botsgo)
    rr = canvas(bgc, (1, w))
    while len(merge(outpi)) < h:
        idx = randint(0, len(outpi) - 1)
        outpi = outpi[:idx] + (rr,) + outpi[idx:]
        outpo = outpo[:idx] + (rr,) + outpo[idx:]
    gi = merge(outpi)
    go = merge(outpo)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_868de0fa(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 7))    
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    num = unifint(diff_lb, diff_ub, (1, 9))
    indss = asindices(gi)
    maxtrials = 4 * num
    tr = 0
    succ = 0
    while succ < num and tr <= maxtrials:
        if len(indss) == 0:
            break
        oh = randint(3, 8)
        ow = oh
        subs = totuple(sfilter(indss, lambda ij: ij[0] < h - oh and ij[1] < w - ow))
        if len(subs) == 0:
            tr += 1
            continue
        loci, locj = choice(subs)
        obj = frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})
        bd = backdrop(obj)
        col = choice(remcols)
        if bd.issubset(indss):
            gi = fill(gi, col, box(bd))
            if oh % 2 == 1:
                go = fill(go, 7, bd)
            else:
                go = fill(go, 2, bd)
            go = fill(go, col, box(bd))
            succ += 1
            indss = (indss - bd) - outbox(bd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_40853293(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nlines = unifint(diff_lb, diff_ub, (2, min(8, (h*w)//2)))
    nhorilines = randint(1, nlines - 1)
    nvertilines = nlines - nhorilines
    ilocs = interval(0, h, 1)
    ilocs = sample(ilocs, min(nhorilines, len(ilocs)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in ilocs:
        llen = unifint(diff_lb, diff_ub, (2, w - 1))
        js = randint(0, w - llen)
        je = js + llen - 1
        a = (ii, js)
        b = (ii, je)
        hln = connect(a, b)
        col = choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, {a, b})
        go = fill(go, col, hln)
    jlocs = interval(0, w, 1)
    gim = dmirror(gi)
    jlocs = sfilter(jlocs, lambda j: sum(1 for e in gim[j] if e == bgc) > 1)
    nvertilines = min(nvertilines, len(jlocs))
    jlocs = sample(jlocs, nvertilines)
    for jj in jlocs:
        jcands = [idx for idx, e in enumerate(gim[jj]) if e == bgc]
        kk = len(jcands)
        locopts = interval(0, kk, 1)
        llen = unifint(diff_lb, diff_ub, (2, kk))
        sp = randint(0, kk - llen)
        ep = sp + llen - 1
        sp = jcands[sp]
        ep = jcands[ep]
        a = (sp, jj)
        b = (ep, jj)
        vln = connect(a, b)
        col = choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, {a, b})
        go = fill(go, col, vln)
    return {'input': gi, 'output': go}


def generate_6e19193c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    dirs = (
    ((0, 0), (-1, -1)),
    ((0, 1), (-1, 1)),
    ((1, 0), (1, -1)),
    ((1, 1), (1, 1))
    )
    base = ((0, 0), (1, 0), (0, 1), (1, 1))
    candsi = [
    set(base) - {dr[0]} for dr in dirs
    ]
    candso = [
    (set(base) | shoot(dr[0], dr[1])) - {dr[0]} for dr in dirs
    ]
    cands = list(zip(candsi, candso))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    fullinds = asindices(gi)
    inds = asindices(canvas(-1, (h, w)))
    kk, tr = 0, 0
    maxtrials = num * 4
    while kk < num and tr < maxtrials:
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        obji, objo = choice(cands)
        obji = shift(obji, loc)
        objo = shift(objo, loc)
        objo = objo & fullinds
        if objo.issubset(inds) and obji.issubset(objo):
            col = choice(remcols)
            gi = fill(gi, col, obji)
            go = fill(go, col, objo)
            inds = (inds - objo) - mapply(dneighbors, obji)
            kk += 1
        tr += 1
    return {'input': gi, 'output': go}


def generate_8731374e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    inh = randint(5, h - 2)
    inw = randint(5, w - 2)
    bgc, fgc = sample(cols, 2)
    num = unifint(diff_lb, diff_ub, (1, min(inh, inw)))
    mat = canvas(bgc, (inh - 2, inw - 2))
    tol = lambda g: list(list(e) for e in g)
    tot = lambda g: tuple(tuple(e) for e in g)
    mat = fill(mat, fgc, connect((0, 0), (num - 1, num - 1)))
    mat = tol(mat)
    shuffle(mat)
    mat = tol(dmirror(tot(mat)))
    shuffle(mat)
    mat = dmirror(tot(mat))
    sgi = paint(canvas(bgc, (inh, inw)), shift(asobject(mat), (1, 1)))
    inds = ofcolor(sgi, fgc)
    lins = mapply(fork(combine, vfrontier, hfrontier), inds)
    go = fill(sgi, fgc, lins)
    numci = unifint(diff_lb, diff_ub, (3, 10))
    numc = 13 - numci
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = asindices(c)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(c, obj)
    loci = randint(1, h - inh - 1)
    locj = randint(1, w - inw - 1)
    loc = (loci, locj)
    plcd = shift(asobject(sgi), loc)
    gi = paint(gi, plcd)
    a, b = ulcorner(plcd)
    c, d = lrcorner(plcd)
    p1 = choice(totuple(connect((a - 1, b), (a - 1, d))))
    p2 = choice(totuple(connect((a, b - 1), (c, b - 1))))
    p3 = choice(totuple(connect((c + 1, b), (c + 1, d))))
    p4 = choice(totuple(connect((a, d + 1), (c, d + 1))))
    remcols = remove(bgc, ccols)
    fixobj = {
        (choice(remcols), p1), (choice(remcols), p2),
        (choice(remcols), p3), (choice(remcols), p4)
    }
    gi = paint(gi, fixobj)
    return {'input': gi, 'output': go}


def generate_cce03e0d(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))    
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nred = unifint(diff_lb, diff_ub, (1, h * w - 1))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, nred)))
    ncells = unifint(diff_lb, diff_ub, (1, h * w - nred))
    ccols = sample(cols, ncols)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    reds = sample(totuple(inds), nred)
    reminds = difference(inds, reds)
    gi = fill(gi, 2, reds)
    rest = sample(totuple(reminds), ncells)
    rest = {(choice(ccols), ij) for ij in rest}
    gi = paint(gi, rest)
    go = canvas(0, (h**2, w**2))
    locs = apply(rbind(multiply, (h, w)), reds)
    res = mapply(lbind(shift, asobject(gi)), locs)
    go = paint(go, res)
    return {'input': gi, 'output': go}


def generate_f9012d9b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)    
    hp = unifint(diff_lb, diff_ub, (2, 10))
    wp = unifint(diff_lb, diff_ub, (2, 10))
    srco = canvas(0, (hp, wp))
    inds = asindices(srco)
    nc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, nc)
    obj = {(choice(ccols), ij) for ij in inds}
    srco = paint(srco, obj)
    gi = paint(srco, obj)
    numhp = unifint(diff_lb, diff_ub, (3, 30 // hp))
    numwp = unifint(diff_lb, diff_ub, (3, 30 // wp))
    for k in range(numhp - 1):
        gi = vconcat(gi, srco)
    srco = tuple(e for e in gi)
    for k in range(numwp - 1):
        gi = hconcat(gi, srco)
    hcropfac = randint(0, hp)
    for k in range(hcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    wcropfac = randint(0, wp)
    for k in range(wcropfac):
        gi = gi[:-1]
    gi = dmirror(gi)
    h, w = shape(gi)
    sgh = unifint(diff_lb, diff_ub, (1, h - hp - 1))
    sgw = unifint(diff_lb, diff_ub, (1, w - wp - 1))
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    loc = (loci, locj)
    shp = (sgh, sgw)
    obj = {loc, decrement(add(loc, shp))}
    obj = backdrop(obj)
    go = subgrid(obj, gi)
    gi = fill(gi, 0, obj)
    mf = choice((
        identity, rot90, rot180, rot270,
        dmirror, vmirror, hmirror, cmirror
    ))
    gi = mf(gi)
    go = mf(go)
    return {'input': gi, 'output': go}


def generate_f8ff0b80(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, min(30, (h * w) // 25)))
    gi = canvas(bgc, (h, w))
    numcells = unifint(diff_lb, diff_ub, (nobjs+1, 36))
    base = asindices(canvas(-1, (6, 6)))
    maxtr = 10
    inds = asindices(gi)
    go = []
    for k in range(nobjs):
        if len(inds) == 0 or numcells < 2:
            break
        numcells = unifint(diff_lb, diff_ub, (nobjs - k, numcells - 1))
        if numcells == 0:
            break
        sp = choice(totuple(base))
        shp = {sp}
        reminds = remove(sp, base)
        for kk in range(numcells - 1):
            shp.add(choice(totuple((reminds - shp) & mapply(neighbors, shp))))
        shp = normalize(shp)
        validloc = False
        rems = sfilter(inds, lambda ij: ij[0] <= h - height(shp) and ij[1] <= w - width(shp))
        if len(rems) == 0:
            break
        loc = choice(totuple(rems))
        tr = 0
        while not validloc and tr < maxtr:
            loc = choice(totuple(inds))
            validloc = shift(shp, loc).issubset(inds)
            tr += 1
        if validloc:
            plcd = shift(shp, loc)
            col = choice(remcols)
            go.append(col)
            inds = (inds - plcd) - mapply(neighbors, plcd)
            gi = fill(gi, col, plcd)
    go = dmirror((tuple(go),))
    return {'input': gi, 'output': go}


def generate_e21d9049(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ph = unifint(diff_lb, diff_ub, (2, 9))
    pw = unifint(diff_lb, diff_ub, (2, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    hbar = frozenset({(choice(remcols), (k, 0)) for k in range(ph)})
    wbar = frozenset({(choice(remcols), (0, k)) for k in range(pw)})
    locih = randint(0, h - ph)
    locjh = randint(0, w - 1)
    loch = (locih, locjh)
    locjw = randint(0, w - pw)
    lociw = randint(0, h - 1)
    locw = (lociw, locjw)
    canv = canvas(bgc, (h, w))
    hbar = shift(hbar, loch)
    wbar = shift(wbar, locw)
    cp = (lociw, locjh)
    col = choice(remcols)
    hbard = extract(hbar, lambda cij: abs(cij[1][0] - lociw) % ph == 0)[1]
    hbar = sfilter(hbar, lambda cij: abs(cij[1][0] - lociw) % ph != 0) | {(col, hbard)}
    wbard = extract(wbar, lambda cij: abs(cij[1][1] - locjh) % pw == 0)[1]
    wbar = sfilter(wbar, lambda cij: abs(cij[1][1] - locjh) % pw != 0) | {(col, wbard)}
    gi = paint(canv, hbar | wbar)
    go = paint(canv, hbar | wbar)
    for k in range(h//ph + 1):
        go = paint(go, shift(hbar, (k*ph, 0)))
        go = paint(go, shift(hbar, (-k*ph, 0)))
    for k in range(w//pw + 1):
        go = paint(go, shift(wbar, (0, k*pw)))
        go = paint(go, shift(wbar, (0, -k*pw)))
    return {'input': gi, 'output': go}


def generate_d4f3cd78(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (3, h//3*2))
    iw = unifint(diff_lb, diff_ub, (3, w//3*2))
    loci = randint(1, h - ih - 1)
    locj = randint(1, w - iw - 1)
    crns = frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)})
    fullcrns = corners(crns)
    bx = box(crns)
    opts = bx - fullcrns
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    nholes = unifint(diff_lb, diff_ub, (1, len(opts)))
    holes = sample(totuple(opts), nholes)
    gi = fill(c, fgc, bx - set(holes))
    bib = backdrop(inbox(bx))
    go = fill(gi, 8, bib)
    A, B = ulcorner(bib)
    C, D = lrcorner(bib)
    f1 = lambda idx: 1 if idx > C else (-1 if idx < A else 0)
    f2 = lambda idx: 1 if idx > D else (-1 if idx < B else 0)
    f = lambda d: shoot(d, (f1(d[0]), f2(d[1])))
    res = mapply(f, set(holes))
    go = fill(go, 8, res)
    return {'input': gi, 'output': go}


def generate_9d9215db(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (5, 14))
    w = unifint(diff_lb, diff_ub, (5, 14))
    h = h * 2 + 1
    w = w * 2 + 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ub = min(h, w)//4
    nrings = unifint(diff_lb, diff_ub, (1, ub))
    onlinesbase = tuple([(2*k+1, 2*k+1) for k in range(ub)])
    onlines = sample(onlinesbase, nrings)
    onlines = {(choice(remcols), ij) for ij in onlines}
    gi = canvas(bgc, (h, w))
    gi = paint(gi, onlines)
    linsbase = apply(rbind(add, (0, 2)), onlinesbase[:-1])
    nlines = unifint(diff_lb, diff_ub, (1, len(linsbase)))
    linesps = sample(linsbase, nlines)
    colors = [choice(remcols) for k in range(nlines)]
    dots = {(col, ij) for col, ij in zip(colors, linesps)}
    dots2 = {(col, ij[::-1]) for col, ij in zip(colors, linesps)}
    gi = paint(gi, dots | dots2)
    ff = lambda ij: ij[1] % 2 == 1
    ff2 = lambda ij: ij[0] % 2 == 1
    linesps2 = tuple(x[::-1] for x in linesps)
    lines = tuple(sfilter(connect(ij, (ij[0], w - ij[1] - 1)), ff) for ij in linesps)
    lines2 = tuple(sfilter(connect(ij, (h - ij[0] - 1, ij[1])), ff2) for ij in linesps2)
    lines = merge({recolor(col, l1 | l2) for col, (l1, l2) in zip(colors, zip(lines, lines2))})
    gobase = paint(gi, lines)
    go = paint(gobase, merge(fgpartition(vmirror(gobase))))
    go = paint(go, merge(fgpartition(hmirror(gobase))))
    go = paint(go, merge(fgpartition(vmirror(hmirror(gobase)))))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_0ca9ddb6(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4, 6, 7, 8))
    xi = {(8, (0, 0))}
    xo = {(8, (0, 0))}
    ai = {(6, (0, 0))}
    ao = {(6, (0, 0))}
    bi = {(2, (1, 1))}
    bo = {(2, (1, 1))} | recolor(4, ineighbors((1, 1)))
    ci = {(1, (1, 1))}
    co = {(1, (1, 1))} | recolor(7, dneighbors((1, 1)))
    arr = ((ai, ao), (bi, bo), (ci, co), (xi, xo))    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 4))
    maxtr = 5 * nobjs
    tr = 0
    succ = 0
    bgc = choice(cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        ino, outo = choice(arr)
        loc = choice(totuple(inds))
        oplcd = shift(outo, loc)
        oplcdi = toindices(oplcd)
        if oplcdi.issubset(inds):
            succ += 1
            gi = paint(gi, shift(ino, loc))
            go = paint(go, oplcd)
            inds = inds - oplcdi
        tr += 1
    return {'input': gi, 'output': go}


def generate_5521c0d9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    inds = interval(0, w, 1)
    nobjs = unifint(diff_lb, diff_ub, (1, w//3))
    speps = sample(inds, nobjs*2)
    while 0 in speps or w - 1 in speps:
        nobjs = unifint(diff_lb, diff_ub, (1, w//3))
        speps = sample(inds, nobjs*2)
    speps = sorted(speps)
    starts = speps[::2]
    ends = speps[1::2]
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    forb = -1
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    forb = -1
    for sp, ep in zip(starts, ends):
        col = choice(remove(forb, ccols))
        forb = col
        hdev = unifint(diff_lb, diff_ub, (0, h//2))
        hei = choice((hdev, h - hdev))
        hei = min(max(1, hei), h - 1)
        ulc = (h - hei, sp)
        lrc = (h - 1, ep)
        obj = backdrop(frozenset({ulc, lrc}))
        gi = fill(gi, col, obj)
        go = fill(go, col, shift(obj, (-hei, 0)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_e3497940(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    bgc, barc = sample(cols, 2)
    remcols = remove(barc, remove(bgc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    nlinesocc = unifint(diff_lb, diff_ub, (1, h))
    lopts = interval(0, h, 1)
    linesocc = sample(lopts, nlinesocc)
    rs = canvas(bgc, (h, w))
    ls = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for idx in linesocc:
        j = unifint(diff_lb, diff_ub, (1, w - 1))
        obj = [(choice(ccols), (idx, jj)) for jj in range(j)]
        go = paint(go, obj)
        slen = randint(1, j)
        obj2 = obj[:slen]
        if choice((True, False)):
            obj, obj2 = obj2, obj
        rs = paint(rs, obj)
        ls = paint(ls, obj2)
    gi = hconcat(hconcat(vmirror(ls), canvas(barc, (h, 1))), rs)
    go = vmirror(go)
    return {'input': gi, 'output': go}


def generate_6cdd2623(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    linc = choice(remcols)
    remcols = remove(linc, remcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, 7))
    noisecols = sample(remcols, nnoisecols)
    c = canvas(bgc, (h, w))
    ininds = totuple(shift(asindices(canvas(-1, (h-2, w-1))), (1, 1)))
    fixinds = sample(ininds, nnoisecols)
    fixobj = {(col, ij) for col, ij in zip(list(noisecols), fixinds)}
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, fixobj)
    nnoise = unifint(diff_lb, diff_ub, (1, (h * w - nnoisecols) // 3))
    noise = sample(totuple(asindices(c) - set(fixinds)), nnoise)
    noise = {(choice(remcols), ij) for ij in noise}
    gi = paint(gi, noise)
    ilocs = interval(1, h - 1, 1)
    jlocs = interval(1, w - 1, 1)
    aa, bb = sample((0, 1), 2)
    nilocs = unifint(diff_lb, diff_ub, (aa, (h - 2) // 2))
    njlocs = unifint(diff_lb, diff_ub, (bb, (w - 2) // 2))
    ilocs = sample(ilocs, nilocs)
    jlocs = sample(jlocs, njlocs)
    for ii in ilocs:
        gi = fill(gi, linc, {(ii, 0)})
        gi = fill(gi, linc, {(ii, w - 1)})
        go = fill(go, linc, connect((ii, 0), (ii, w - 1)))
    for jj in jlocs:
        gi = fill(gi, linc, {(0, jj)})
        gi = fill(gi, linc, {(h - 1, jj)})
        go = fill(go, linc, connect((0, jj), (h - 1, jj)))
    return {'input': gi, 'output': go}


def generate_dc433765(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, src = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    if choice((True, False)):
        opts = {(ii, 0) for ii in range(h - 2)} | {(0, jj) for jj in range(1, w - 2, 1)}
        opts = tuple([inds & shoot(src, (1, 1)) for src in opts])
        opts = order(opts, size)
        k = len(opts)
        opt = unifint(diff_lb, diff_ub, (0, k - 1))
        ln = order(opts[opt], first)
        epi = unifint(diff_lb, diff_ub, (2, len(ln) - 1))
        ep = ln[epi]
        ln = ln[:epi-1][::-1]
        spi = unifint(diff_lb, diff_ub, (0, len(ln) - 1))
        sp = ln[spi]
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (1, 1))})
        go = fill(go, 4, {ep})
    else:
        loci = randint(0, h - 1)
        objw = unifint(diff_lb, diff_ub, (3, w))
        locj1 = randint(0, w - objw)
        locj2 = locj1 + objw - 1
        sp = (loci, locj1)
        ep = (loci, locj2)
        gi = fill(gi, src, {sp})
        gi = fill(gi, 4, {ep})
        go = fill(go, src, {add(sp, (0, 1))})
        go = fill(go, 4, {ep})
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_d2abd087(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(difference(cols, (1, 2)))
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        opts = asindices(canvas(-1, (5, 5)))
        sp = choice(totuple(opts))
        opts = remove(sp, opts)
        lb = unifint(diff_lb, diff_ub, (1, 5))
        lopts = interval(lb, 6, 1)
        ubi = unifint(diff_lb, diff_ub, (1, 5))
        ub = 12 - ubi
        uopts = interval(7, ub + 1, 1)
        if choice((True, False)):
            numcells = 6
        else:
            numcells = choice(lopts + uopts)
        obj = {sp}
        for k in range(numcells - 1):
            obj.add(choice(totuple((opts - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        loc = choice(totuple(inds))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            gi = fill(gi, choice(remcols), plcd)
            go = fill(go, 1 + (len(obj) == 6), plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        tr += 1
    return {'input': gi, 'output': go}


def generate_88a10436(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (0, 2))
    objw = unifint(diff_lb, diff_ub, (0 if objh > 0 else 1, 2))
    objh = objh * 2 + 1
    objw = objw * 2 + 1
    bb = asindices(canvas(-1, (objh, objw)))
    sp = (objh // 2, objw // 2)
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (max(objh, objw), objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    while height(obj) != objh or width(obj) != objw:
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    bgc, fgc = sample(cols, 2)
    remcols = remove(bgc, remove(fgc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    obj = {(choice(ccols), ij) for ij in obj}
    obj = normalize(obj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    loc = (loci, locj)
    plcd = shift(obj, loc)
    gi = paint(gi, plcd)
    go = paint(go, plcd)
    inds = (asindices(gi) - toindices(plcd)) - mapply(neighbors, toindices(plcd))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * ncells)))
    maxtrials = 4 * nobjs
    tr = 0
    succ = 0
    while succ < nobjs and tr <= maxtrials:
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            go = paint(go, plcd)
            gi = fill(gi, fgc, {center(plcdi)})
            succ += 1
            inds = (inds - plcdi) - mapply(dneighbors, plcdi)
        tr += 1
    return {'input': gi, 'output': go}


def generate_05f2a901(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    objh = unifint(diff_lb, diff_ub, (2, min(w//2, h//2)))
    objw = unifint(diff_lb, diff_ub, (objh, w//2))
    bb = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncells = unifint(diff_lb, diff_ub, (objh + objw, objh * objw))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bb - obj) & mapply(dneighbors, obj))))
    if height(obj) * width(obj) == len(obj):
        obj = remove(choice(totuple(obj)), obj)
    obj = normalize(obj)
    objh, objw = shape(obj)
    loci = unifint(diff_lb, diff_ub, (3, h - objh))
    locj = unifint(diff_lb, diff_ub, (0, w - objw))
    loc = (loci, locj)
    bgc, fgc, destc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(obj, loc)
    gi = fill(gi, fgc, obj)
    sqd = randint(1, min(w, loci - 1))
    locisq = randint(0, loci-sqd-1)
    locjsq = randint(locj-sqd+1, locj+objw-1)
    locsq = (locisq, locjsq)
    sq = backdrop({(locisq, locjsq), (locisq+sqd-1, locjsq+sqd-1)})
    gi = fill(gi, destc, sq)
    go = fill(go, destc, sq)
    while len(obj & sq) == 0:
        obj = shift(obj, (-1, 0))
    obj = shift(obj, (1, 0))
    go = fill(go, fgc, obj)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_928ad970(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    ih = unifint(diff_lb, diff_ub, (9, h))
    iw = unifint(diff_lb, diff_ub, (9, w))
    bgc, linc, dotc = sample(cols, 3)
    loci = randint(0, h - ih)
    locj = randint(0, w - iw)
    ulc = (loci, locj)
    lrc = (loci + ih - 1, locj + iw - 1)
    dot1 = choice(totuple(connect(ulc, (loci + ih - 1, locj)) - {ulc, (loci + ih - 1, locj)}))
    dot2 = choice(totuple(connect(ulc, (loci, locj + iw - 1)) - {ulc, (loci, locj + iw - 1)}))
    dot3 = choice(totuple(connect(lrc, (loci + ih - 1, locj)) - {lrc, (loci + ih - 1, locj)}))
    dot4 = choice(totuple(connect(lrc, (loci, locj + iw - 1)) - {lrc, (loci, locj + iw - 1)}))
    a, b = sorted(sample(interval(loci + 2, loci + ih - 2, 1), 2))
    while a + 1 == b:
        a, b = sorted(sample(interval(loci + 2, loci + ih - 2, 1), 2))
    c, d = sorted(sample(interval(locj + 2, locj + iw - 2, 1), 2))
    while c + 1 == d:
        c, d = sorted(sample(interval(locj + 2, locj + iw - 2, 1), 2))
    sp = box(frozenset({(a, c), (b, d)}))
    bx = {dot1, dot2, dot3, dot4}
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, dotc, bx)
    gi = fill(gi, linc, sp)
    go = fill(gi, linc, inbox(bx))
    return {'input': gi, 'output': go}


def generate_f8b3ba0a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 5))
    w = unifint(diff_lb, diff_ub, (1, 5))
    nh = unifint(diff_lb, diff_ub, (3, 29 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (3, 29 // (w + 1)))
    fullh = (h + 1) * nh + 1
    fullw = (w + 1) * nw + 1
    fullbgc, bgc = sample(cols, 2)
    remcols = remove(fullbgc, remove(bgc, cols))
    shp = shift(asindices(canvas(-1, (h, w))), (1, 1))
    gi = canvas(fullbgc, (fullh, fullw))
    locs = set()
    for a in range(nh):
        for b in range(nw):
            loc = (a * (h + 1), b * (w + 1))
            locs.add(loc)
            gi = fill(gi, bgc, shift(shp, loc))
    numc = unifint(diff_lb, diff_ub, (1, (nh * nw) // 2 - 1))
    stack = []
    nn = numc + 1
    ncols = 0
    while nn > 1 and numc > 0 and len(remcols) > 0:
        nn3 = int(0.5 * (8 * numc + 1) ** 0.5 - 1)
        nn = min(max(1, nn3), nn - 1)
        col = choice(remcols)
        remcols = remove(col, remcols)
        numc -= nn
        stack.append((col, nn))
    go = dmirror((tuple(c for c, nn in stack),))
    for col, nn in stack:
        slocs = sample(totuple(locs), nn)
        gi = fill(gi, col, mapply(lbind(shift, shp), slocs))
        locs = locs - set(slocs)
    return {'input': gi, 'output': go}


def generate_fcb5c309(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, dotc, sqc = sample(cols, 3)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    maxtr = 4 * numsq
    tr = 0
    succ = 0
    numcells = None
    take = False
    while tr < maxtr and succ < numsq:
        oh = randint(3, 7)
        ow = randint(3, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(sq)
        if bd.issubset(inds):
            gi = fill(gi, sqc, sq)
            ib = backdrop(inbox(sq))
            if numcells is None:
                numcells = unifint(diff_lb, diff_ub, (1, len(ib)))
                cells = sample(totuple(ib), numcells)
                take = True
            else:
                nc = unifint(diff_lb, diff_ub, (0, min(max(0, numcells - 1), len(ib))))
                cells = sample(totuple(ib), nc)
            gi = fill(gi, dotc, cells)
            if take:
                go = replace(subgrid(sq, gi), sqc, dotc)
                take = False
            inds = (inds - bd) - outbox(bd)
            succ += 1
        tr += 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, len(inds) // 2 - 1)))
    noise = sample(totuple(inds), nnoise)
    gi = fill(gi, dotc, noise)
    return {'input': gi, 'output': go}


def generate_54d9e175(diff_lb: float, diff_ub: float) -> dict:
    cols = (0, 5)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (1, 31 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (1 if nh > 1 else 2, 31 // (w + 1)))
    fullh = (h + 1) * nh - 1
    fullw = (w + 1) * nw - 1
    linc, bgc = sample(cols, 2)
    gi = canvas(linc, (fullh, fullw))
    go = canvas(linc, (fullh, fullw))
    obj = asindices(canvas(bgc, (h, w)))
    for a in range(nh):
        for b in range(nw):
            plcd = shift(obj, (a * (h + 1), b * (w + 1)))
            icol = randint(1, 4)
            ocol = icol + 5
            gi = fill(gi, bgc, plcd)
            go = fill(go, ocol, plcd)
            dot = choice(totuple(plcd))
            gi = fill(gi, icol, {dot})
    return {'input': gi, 'output': go}


def generate_7f4411dc(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 15))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < nsq:
        oh = randint(2, 6)
        ow = randint(2, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        loci, locj = loc
        obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        obj = shift(obj, loc)
        if obj.issubset(inds):
            go = fill(go, fgc, obj)
            succ += 1
            inds = (inds - obj) - outbox(obj)
        tr += 1
    inds = ofcolor(go, bgc)
    nnoise = unifint(diff_lb, diff_ub, (0, len(inds) // 2 - 1))
    gi = tuple(e for e in go)
    for k in range(nnoise):
        loc = choice(totuple(inds))
        inds = inds - dneighbors(loc)
        gi = fill(gi, fgc, {loc})
    return {'input': gi, 'output': go}


def generate_67385a82(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(0, remove(8, interval(0, 10, 1)))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col = choice(cols)
    gi = canvas(0, (h, w))
    inds = totuple(asindices(gi))
    ncd = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    nc = choice((ncd, len(inds) - ncd))
    nc = min(max(1, nc), len(inds) - 1)
    locs = sample(inds, nc)
    gi = fill(gi, col, locs)
    objs = objects(gi, T, F, F)
    rems = toindices(merge(sizefilter(colorfilter(objs, col), 1)))
    blues = difference(ofcolor(gi, col), rems)
    go = fill(gi, 8, blues)
    return {'input': gi, 'output': go}


def generate_d6ad076f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    inh = unifint(diff_lb, diff_ub, (3, h))
    inw = unifint(diff_lb, diff_ub, (3, w))
    bgc, c1, c2 = sample(cols, 3)
    itv = interval(0, inh, 1)
    loci2i = unifint(diff_lb, diff_ub, (2, inh - 1))
    loci2 = itv[loci2i]
    itv = itv[:loci2i-1][::-1]
    loci1i = unifint(diff_lb, diff_ub, (0, len(itv) - 1))
    loci1 = itv[loci1i]
    cp = randint(1, inw - 2)
    ajs = randint(0, cp - 1)
    aje = randint(cp + 1, inw - 1)
    bjs = randint(0, cp - 1)
    bje = randint(cp + 1, inw - 1)
    obja = backdrop(frozenset({(0, ajs), (loci1, aje)}))
    objb = backdrop(frozenset({(loci2, bjs), (inh - 1, bje)}))
    c = canvas(bgc, (inh, inw))
    c = fill(c, c1, obja)
    c = fill(c, c2, objb)
    obj = asobject(c)
    loci = randint(0, h - inh)
    locj = randint(0, w - inw)
    loc = (loci, locj)
    obj = shift(obj, loc)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, obj)
    midobj = backdrop(frozenset({(loci1 + 1, max(ajs, bjs) + 1), (loci2 - 1, min(aje, bje) - 1)}))
    go = fill(gi, 8, shift(midobj, loc))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_e48d4e1a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    inds = asindices(canvas(-1, (loci, locj)))
    maxn = min(min(h - loci - 1, w - locj - 1), len(inds))
    nn = unifint(diff_lb, diff_ub, (1, maxn))
    ss = sample(totuple(inds), nn)
    bgc, fgc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, fgc, hfrontier((loci, 0)) | vfrontier((0, locj)))
    gi = fill(gi, dotc, ss)
    go = fill(go, fgc, hfrontier((loci + nn, 0)) | vfrontier((0, locj + nn)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_a48eeaf7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ih = unifint(diff_lb, diff_ub, (2, h//2))
    iw = unifint(diff_lb, diff_ub, (2, w//2))
    loci = randint(2, h - ih - 2)
    locj = randint(2, w - iw - 2)
    bgc, sqc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    sq = backdrop(frozenset({(loci, locj), (loci + ih - 1, locj + iw - 1)}))
    A = [(x, locj - 1) for x in interval(loci, loci + ih, 1)]
    Ap = [(x, randint(0, locj - 2)) for x in interval(loci, loci + ih, 1)]
    B = [(x, locj + iw) for x in interval(loci, loci + ih, 1)]
    Bp = [(x, randint(locj + iw + 1, w - 1)) for x in interval(loci, loci + ih, 1)]
    C = [(loci - 1, x) for x in interval(locj, locj + iw, 1)]
    Cp = [(randint(0, loci - 2), x) for x in interval(locj, locj + iw, 1)]
    D = [(loci + ih, x) for x in interval(locj, locj + iw, 1)]
    Dp = [(randint(loci + ih + 1, h - 1), x) for x in interval(locj, locj + iw, 1)]
    srarr = Ap + Bp + Cp + Dp
    dearr = A + B + C + D
    inds = interval(0, len(srarr), 1)
    num = unifint(diff_lb, diff_ub, (1, len(srarr)))
    locs = sample(inds, num)
    srarr = [e for j, e in enumerate(srarr) if j in locs]
    dearr = [e for j, e in enumerate(dearr) if j in locs]
    gi = fill(gi, sqc, sq)
    go = fill(go, sqc, sq)
    for s, d in zip(srarr, dearr):
        gi = fill(gi, dotc, {s})
        go = fill(go, dotc, {d})
    ncorn = unifint(diff_lb, diff_ub, (0, 4))
    fullinds = asindices(gi)
    if ncorn > 0:
        go = fill(go, dotc, {(loci - 1, locj - 1)})
        cands = shoot((loci - 2, locj - 2), (-1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 1:
        go = fill(go, dotc, {(loci - 1, locj + iw)})
        cands = shoot((loci - 2, locj + iw + 1), (-1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 2:
        go = fill(go, dotc, {(loci + ih, locj - 1)})
        cands = shoot((loci + ih + 1, locj - 2), (1, -1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    if ncorn > 3:
        go = fill(go, dotc, {(loci + ih, locj + iw)})
        cands = shoot((loci + ih + 1, locj + iw + 1), (1, 1)) & fullinds
        locc = choice(totuple(cands))
        gi = fill(gi, dotc, {locc})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_56dc2b01(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    oh = unifint(diff_lb, diff_ub, (1, h))
    ow = unifint(diff_lb, diff_ub, (1, (w - 1) // 2 - 1))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    locj = unifint(diff_lb, diff_ub, (1, w - ow))
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    barlocji = unifint(diff_lb, diff_ub, (0, locj))
    barlocj = locj - barlocji
    barlocj = min(max(0, barlocj), locj - 1)
    gi = fill(gi, 2, connect((0, barlocj), (h - 1, barlocj)))
    go = fill(gi, objc, shift(obj, (loci, barlocj + 1)))
    go = fill(go, 8, connect((0, barlocj + ow + 1), (h - 1, barlocj + ow + 1)))
    gi = fill(gi, objc, shift(obj, (loci, locj)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_1caeab9d(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1,))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    oh = unifint(diff_lb, diff_ub, (1, h//2))
    ow = unifint(diff_lb, diff_ub, (1, w//3))
    bb = asindices(canvas(-1, (oh, ow)))
    sp = choice(totuple(bb))
    obj = {sp}
    bb = remove(sp, bb)
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(0, ncells), oh * ow - 1)
    for k in range(ncells):
        obj.add(choice(totuple((bb - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    numo = unifint(diff_lb, diff_ub, (2, min(8, w // ow))) - 1
    itv = interval(0, w, 1)
    locj = randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    c = canvas(bgc, (h, w))
    gi = fill(c, 1, objp)
    go = fill(c, 1, objp)
    itv = difference(itv, interval(locj, locj + ow, 1))
    for k in range(numo):
        cands = sfilter(itv, lambda j: set(interval(j, j + ow, 1)).issubset(set(itv)))
        if len(cands) == 0:
            break
        locj = choice(cands)
        col = choice(remcols)
        remcols = remove(col, remcols)
        gi = fill(gi, col, shift(obj, (randint(0, h - oh), locj)))
        go = fill(go, col, shift(obj, (loci, locj)))
        itv = difference(itv, interval(locj, locj + ow, 1))
    return {'input': gi, 'output': go}


def generate_b91ae062(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    numc = unifint(diff_lb, diff_ub, (3, min(h * w, min(10, 30 // max(h, w)))))
    ccols = sample(cols, numc)
    c = canvas(-1, (h, w))
    inds = totuple(asindices(c))
    fixinds = sample(inds, numc)
    obj = {(cc, ij) for cc, ij in zip(ccols, fixinds)}
    for ij in difference(inds, fixinds):
        obj.add((choice(ccols), ij))
    gi = paint(c, obj)
    go = upscale(gi, numc - 1)
    return {'input': gi, 'output': go}


def generate_834ec97d(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    loci = unifint(diff_lb, diff_ub, (0, h - 2))
    locjd = unifint(diff_lb, diff_ub, (0, w // 2))
    locj = choice((locjd, w - locjd))
    locj = min(max(0, locj), w - 1)
    loc = (loci, locj)
    bgc, fgc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    gi = fill(c, fgc, {loc})
    go = fill(c, fgc, {add(loc, (1, 0))})
    for jj in range(w//2 + 1):
        go = fill(go, 4, connect((0, locj + 2 * jj), (loci, locj + 2 * jj)))
        go = fill(go, 4, connect((0, locj - 2 * jj), (loci, locj - 2 * jj)))
    return {'input': gi, 'output': go}


def generate_a699fb00(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numls = unifint(diff_lb, diff_ub, (1, h - 1))
    opts = interval(0, h, 1)
    locs = sample(opts, numls)
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in locs:
        endidx = unifint(diff_lb, diff_ub, (2, w - 2))
        ofs = unifint(diff_lb, diff_ub, (1, endidx//2)) * 2
        ofs = min(max(2, ofs), endidx)
        startidx = endidx - ofs
        ln = connect((ii, startidx), (ii, endidx))
        go = fill(go, 2, ln)
        sparseln = {(ii, jj) for jj in range(startidx, endidx + 1, 2)}
        go = fill(go, fgc, sparseln)
        gi = fill(gi, fgc, sparseln)
    return {'input': gi, 'output': go}


def generate_91413438(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    maxnb = min(h * w - 1, min(30//h, 30//w))
    minnb = int(0.5 * ((4 * h * w + 1) ** 0.5 - 1)) + 1
    nbi = unifint(diff_lb, diff_ub, (0, maxnb - minnb))
    nb = min(max(minnb, maxnb - nbi), maxnb)
    fgc = choice(cols)
    c = canvas(0, (h, w))
    obj = sample(totuple(asindices(c)), h * w - nb)
    gi = fill(c, fgc, obj)
    go = canvas(0, (h * nb, w * nb))
    for j in range(h * w - nb):
        loc = (j // nb, j % nb)
        go = fill(go, fgc, shift(obj, multiply((h, w), loc)))
    return {'input': gi, 'output': go}


def generate_99fa7670(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    num = unifint(diff_lb, diff_ub, (1, h // 2))
    inds = interval(0, h, 1)
    starts = sorted(sample(inds, num))
    ends = [x - 1 for x in starts[1:]] + [h - 1]
    nc = unifint(diff_lb, diff_ub, (1, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for s, e in zip(starts, ends):
        col = choice(ccols)
        locj = randint(0, w - 2)
        l1 = connect((s, locj), (s, w - 1))
        l2 = connect((s, w - 1), (e, w - 1))
        gi = fill(gi, col, {(s, locj)})
        go = fill(go, col, l1 | l2)
    return {'input': gi, 'output': go}


def generate_d13f3404(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (3, 15))
    vopts = {(ii, 0) for ii in interval(0, h, 1)}
    hopts = {(0, jj) for jj in interval(1, w, 1)}
    opts = tuple(vopts | hopts)
    num = unifint(diff_lb, diff_ub, (1, len(opts)))
    locs = sample(opts, num)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h*2, w*2))
    inds = asindices(gi)
    for loc in locs:
        ln = tuple(shoot(loc, (1, 1)) & inds)
        locc = choice(ln)
        col = choice(remcols)
        gi = fill(gi, col, {locc})
        go = fill(go, col, shoot(locc, (1, 1)))
    return {'input': gi, 'output': go}


def generate_c3f564a4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    p = unifint(diff_lb, diff_ub, (2, min(9, min(h//3, w//3))))
    fixc = choice(cols)
    remcols = remove(fixc, cols)
    ccols = list(sample(remcols, p))
    shuffle(ccols)
    c = canvas(-1, (h, w))
    baseobj = {(cc, (0, jj)) for cc, jj in zip(ccols, range(p))}
    obj = {c for c in baseobj}
    while rightmost(obj) < 2 * max(w, h):
        obj = obj | shift(obj, (0, p))
    if choice((True, False)):
        obj = mapply(lbind(shift, obj), {(jj, 0) for jj in interval(0, h, 1)})
    else:
        obj = mapply(lbind(shift, obj), {(jj, -jj) for jj in interval(0, h, 1)})
    go = paint(c, obj)
    gi = tuple(e for e in go)
    nsq = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 25)))
    maxtr = 4 * nsq
    tr = 0
    succ = 0
    while succ < nsq and tr < maxtr:
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        tmpg = fill(gi, fixc, backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)})))
        if len(occurrences(tmpg, baseobj)) > 1 and len([r for r in tmpg if fixc not in r]) > 0 and len([r for r in dmirror(tmpg) if fixc not in r]) > 0:
            gi = tmpg
            succ += 1
        tr += 1
    if choice((True, False)):
        gi = rot90(gi)
        go = rot90(go)
    return {'input': gi, 'output': go}


def generate_ecdecbb3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, dotc, linc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    nl = unifint(diff_lb, diff_ub, (1, h//4))
    inds = interval(0, h, 1)
    locs = []
    for k in range(nl):
        if len(inds) == 0:
            break
        idx = choice(inds)
        locs.append(idx)
        inds = remove(idx, inds)
        inds = remove(idx - 1, inds)
        inds = remove(idx + 1, inds)
        inds = remove(idx - 2, inds)
        inds = remove(idx + 2, inds)
    locs = sorted(locs)
    for ii in locs:
        gi = fill(gi, linc, hfrontier((ii, 0)))
    iopts = difference(difference(difference(interval(0, h, 1), locs), apply(increment, locs)), apply(decrement, locs))
    jopts = interval(0, w, 1)
    ndots = unifint(diff_lb, diff_ub, (1, min(len(iopts), w // 2)))
    dlocs = []
    for k in range(ndots):
        if len(iopts) == 0 or len(jopts) == 0:
            break
        loci = choice(iopts)
        locj = choice(jopts)
        dlocs.append((loci, locj))
        jopts = remove(locj, jopts)
        jopts = remove(locj+1, jopts)
        jopts = remove(locj-1, jopts)
    go = gi
    for d in dlocs:
        loci, locj = d
        if loci < min(locs):
            go = fill(go, dotc, connect(d, (min(locs), locj)))
            go = fill(go, linc, neighbors((min(locs), locj)))
        elif loci > max(locs):
            go = fill(go, dotc, connect(d, (max(locs), locj)))
            go = fill(go, linc, neighbors((max(locs), locj)))
        else:
            sp = [e for e in locs if e < loci][-1]
            ep = [e for e in locs if e > loci][0]
            go = fill(go, dotc, connect((sp, locj), (ep, locj)))
            go = fill(go, linc, neighbors((sp, locj)))
            go = fill(go, linc, neighbors((ep, locj)))
        gi = fill(gi, dotc, {d})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_ac0a08a4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    num = unifint(diff_lb, diff_ub, (1, min(min(9, h * w - 2), min(30//h, 30//w))))
    bgc = choice(cols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = sample(totuple(inds), num)
    remcols = remove(bgc, cols)
    obj = {(col, loc) for col, loc in zip(sample(remcols, num), locs)}
    gi = paint(c, obj)
    go = upscale(gi, num)
    return {'input': gi, 'output': go}


def generate_22168020(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    num = unifint(diff_lb, diff_ub, (1, min(9, (h * w) // 10)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 6 * num
    inds = asindices(gi)
    while tr < maxtr and succ < num:
        d = unifint(diff_lb, diff_ub, (2, 5))
        oh = d + 1
        ow = 2 * d
        if len(inds) == 0:
            tr += 1
            continue
        loc = choice(totuple(inds))
        loci, locj = loc
        io1 = connect(loc, (loci + d - 1, locj + d - 1))
        io2 = connect((loci, locj + ow - 1), (loci + d - 1, locj + d))
        io = io1 | io2 | {(loci + d, locj + d - 1), (loci + d, locj + d)}
        oo = merge(sfilter(prapply(connect, io, io), hline))
        mf = choice((identity, dmirror, cmirror, hmirror, vmirror))
        io = mf(io)
        oo = mf(oo)
        col = choice(remcols)
        if oo.issubset(inds):
            gi = fill(gi, col, io)
            go = fill(go, col, oo)
            succ += 1
            inds = inds - oo
            remcols = remove(col, remcols)
        tr += 1
    return {'input': gi, 'output': go}


def generate_ff805c23(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = h
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    remcols = sample(remcols, numcols)
    canv = canvas(bgc, (h, w))
    nc = unifint(diff_lb, diff_ub, (1, h * w))
    bx = asindices(canv)
    obj = {(choice(remcols), choice(totuple(bx)))}
    for kk in range(nc - 1):
        dns = mapply(neighbors, toindices(obj))
        ch = choice(totuple(bx & dns))
        obj.add((choice(remcols), ch))
        bx = bx - {ch}
    gi = paint(canv, obj)
    tr = sfilter(asobject(dmirror(gi)), lambda cij: cij[1][1] >= cij[1][0])
    gi = paint(gi, tr)
    gi = hconcat(gi, vmirror(gi))
    gi = vconcat(gi, hmirror(gi))
    locidev = unifint(diff_lb, diff_ub, (1, 2*h))
    locjdev = unifint(diff_lb, diff_ub, (1, w))
    loci = 2*h - locidev
    locj = w - locjdev
    loci2 = unifint(diff_lb, diff_ub, (loci, 2*h - 1))
    locj2 = unifint(diff_lb, diff_ub, (locj, w - 1))
    bd = backdrop(frozenset({(loci, locj), (loci2, locj2)}))
    go = subgrid(bd, gi)
    gi = fill(gi, 0, bd)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_4093f84a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    loci1, loci2 = sorted(sample(interval(2, h - 2, 1), 2))
    bgc, barc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    for ii in range(loci1, loci2+1, 1):
        gi = fill(gi, barc, connect((ii, 0), (ii, w - 1)))
    go = tuple(e for e in gi)
    opts = interval(0, w, 1)
    num1 = unifint(diff_lb, diff_ub, (1, w // 2))
    num2 = unifint(diff_lb, diff_ub, (1, w // 2))
    locs1 = sample(opts, num1)
    locs2 = sample(opts, num2)
    for l1 in locs1:
        k = unifint(diff_lb, diff_ub, (1, loci1 - 1))
        locsx = sample(interval(0, loci1, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l1), locsx))
        go = fill(go, barc, connect((loci1 - 1, l1), (loci1 - k, l1)))
    for l2 in locs2:
        k = unifint(diff_lb, diff_ub, (1, h - loci2 - 2))
        locsx = sample(interval(loci2+1, h, 1), k)
        gi = fill(gi, dotc, apply(rbind(astuple, l2), locsx))
        go = fill(go, barc, connect((loci2 + 1, l2), (loci2 + k, l2)))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_760b3cac(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    objL = frozenset({(0, 0), (1, 0), (1, 1), (1, 2), (2, 1)})
    objR = vmirror(objL)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (3, 14))
    w = 2 * w + 1
    bgc, objc, indc = sample(cols, 3)
    objh = unifint(diff_lb, diff_ub, (1, h - 3))
    objw = unifint(diff_lb, diff_ub, (1, w // 6))
    objw = 2 * objw + 1
    c = canvas(-1, (objh, objw))
    gi = canvas(bgc, (h, w))
    if choice((True, False)):
        obj = objL
        sgn = -1
    else:
        obj = objR
        sgn = 1
    gi = fill(gi, indc, shift(obj, (h - 3, w//2 - 1)))
    inds = asindices(c)
    sp = choice(totuple(inds))
    objx = {sp}
    numcd = unifint(diff_lb, diff_ub, (0, (objh * objw) // 2))
    numc = choice((numcd, objh * objw - numcd))
    numc = min(max(1, numc), objh * objw)
    for k in range(numc - 1):
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    while width(objx) != objw:
        objx.add(choice(totuple((inds - objx) & mapply(neighbors, objx))))
    objx = normalize(objx)
    objh, objw = shape(objx)
    loci = randint(0, h - 3 - objh)
    locj = w//2 - objw//2
    loc = (loci, locj)
    plcd = shift(objx, loc)
    gi = fill(gi, objc, plcd)
    objx2 = vmirror(plcd)
    plcd2 = shift(objx2, (0, objw * sgn))
    go = fill(gi, objc, plcd2)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_8efcae92(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sqc, dotc = sample(cols, 3)
    num = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    maxtr = 4 * num
    tr = 0
    gi = canvas(bgc, (h, w))
    go = None
    inds = asindices(gi)
    oho, owo = None, None
    while succ < num and tr < maxtr:
        if oho is None and owo is None:
            oh = randint(2, h - 1)
            ow = randint(2, w - 1)
            oho = oh
            owo = ow
        else:
            ohd = unifint(diff_lb, diff_ub, (0, min(oho, h - 1 - oho)))
            owd = unifint(diff_lb, diff_ub, (0, min(owo, w - 1 - owo)))
            ohd = min(oho, h - 1 - oho) - ohd
            owd = min(owo, w - 1 - owo) - owd
            oh = choice((oho - ohd, oho + ohd))
            ow = choice((owo - owd, owo + owd))
            oh = min(max(2, oh), h - 1)
            ow = min(max(2, ow), w - 1)
        minig = canvas(sqc, (oh, ow))
        mini = asindices(minig)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        if not shift(mini, loc).issubset(inds):
            continue
        succ += 1
        if go is None:
            numdots = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2 - 1))
            nd = numdots
        else:
            nd = unifint(diff_lb, diff_ub, (0, min((oh * ow) // 2 - 1, numdots - 1)))
        locs = sample(totuple(mini), nd)
        minig = fill(minig, dotc, locs)
        if go is None:
            go = minig
        obj = asobject(minig)
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        inds = (inds - toindices(plcd)) - mapply(dneighbors, toindices(plcd))
    return {'input': gi, 'output': go}


def generate_48d8fb45(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < nobjs:
        oh = randint(2, 6)
        ow = randint(2, 6)
        bx = asindices(canvas(-1, (oh, ow)))
        nc = randint(3, oh * ow)
        sp = choice(totuple(bx))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            obj.add(choice(totuple((bx - obj) & mapply(neighbors, obj))))
        if not done:
            done = True
            idx = choice(totuple(obj))
            coll = choice(remcols)
            obj2 = {(coll, idx)}
            obj3 = recolor(choice(remove(coll, remcols)), remove(idx, obj))
            obj = obj2 | obj3
            go = paint(canvas(bgc, shape(obj3)), normalize(obj3))
        else:
            obj = recolor(choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        loc = choice(totuple(locopts))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return {'input': gi, 'output': go}


def generate_8e1813be(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    nbars = unifint(diff_lb, diff_ub, (3, 8))
    ccols = sample(remcols, nbars)
    w = unifint(diff_lb, diff_ub, (nbars+3, 30))
    hmarg = unifint(diff_lb, diff_ub, (2 * nbars, 30 - nbars))
    ccols = list(ccols)
    go = tuple(repeat(col, nbars) for col in ccols)
    gi = tuple(repeat(col, w) for col in ccols)
    r = repeat(bgc, w)
    for k in range(hmarg):
        idx = randint(0, len(go) - 1)
        gi = gi[:idx] + (r,) + gi[idx:]
    h2 = nbars + hmarg
    oh, ow = nbars, nbars
    loci = randint(1, h2 - oh - 2)
    locj = randint(1, w - ow - 2)
    sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, sq)
    gi = fill(gi, bgc, outbox(sq))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_5117e062(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 15))
    tr = 0
    maxtr = 4 * nobjs
    done = False
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < nobjs:
        oh = randint(2, 6)
        ow = randint(2, 6)
        bx = asindices(canvas(-1, (oh, ow)))
        nc = randint(3, oh * ow)
        sp = choice(totuple(bx))
        bx = remove(sp, bx)
        obj = {sp}
        for k in range(nc - 1):
            obj.add(choice(totuple((bx - obj) & mapply(neighbors, obj))))
        if not done:
            done = True
            idx = choice(totuple(obj))
            coll = choice(remcols)
            obj2 = {(coll, idx)}
            coll2 = choice(remove(coll, remcols))
            obj3 = recolor(coll2, remove(idx, obj))
            obj = obj2 | obj3
            go = fill(canvas(bgc, shape(obj)), coll2, normalize(obj))
        else:
            obj = recolor(choice(remcols), obj)
        locopts = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        tr += 1
        if len(locopts) == 0:
            continue
        loc = choice(totuple(locopts))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            gi = paint(gi, plcd)
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
    return {'input': gi, 'output': go}


def generate_f15e1fac(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    nsps = unifint(diff_lb, diff_ub, (1, (w-1) // 2))
    ngps = unifint(diff_lb, diff_ub, (1, (h-1) // 2))
    spsj = sorted(sample(interval(1, w - 1, 1), nsps))
    gpsi = sorted(sample(interval(1, h - 1, 1), ngps))
    ofs = 0
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, {(0, jj) for jj in spsj})
    gi = fill(gi, 2, {(ii, 0) for ii in gpsi})
    go = tuple(e for e in gi)
    for a, b in zip([0] + gpsi, [x - 1 for x in gpsi] + [h - 1]):
        for jj in spsj:
            go = fill(go, linc, connect((a, jj + ofs), (b, jj + ofs)))
        ofs += 1
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_3906de3d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    oh = unifint(diff_lb, diff_ub, (2, h // 2))
    ow = unifint(diff_lb, diff_ub, (3, w - 2))
    bgc, boxc, linc = sample(cols, 3)
    locj = randint(1, w - ow - 1)
    bx = backdrop(frozenset({(0, locj), (oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, boxc, bx)
    rng = range(locj, locj + ow)
    cutoffs = [randint(1, oh - 1) for j in rng]
    for jj, co in zip(rng, cutoffs):
        gi = fill(gi, bgc, connect((co, jj), (oh - 1, jj)))
    numlns = unifint(diff_lb, diff_ub, (1, ow - 1))
    lnlocs = sample(list(rng), numlns)
    go = tuple(e for e in gi)
    for jj, co in zip(rng, cutoffs):
        if jj in lnlocs:
            lineh = randint(1, h - co - 1)
            linei = connect((h - lineh, jj), (h - 1, jj))
            lineo = connect((co, jj), (co + lineh - 1, jj))
            gi = fill(gi, linc, linei)
            go = fill(go, linc, lineo)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_77fdfe62(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 13))
    w = unifint(diff_lb, diff_ub, (1, 13))
    c1, c2, c3, c4, barc, bgc, inc = sample(cols, 7)
    qd = canvas(bgc, (h, w))
    inds = totuple(asindices(qd))
    fullh = 2 * h + 4
    fullw = 2 * w + 4
    n1 = unifint(diff_lb, diff_ub, (1, h * w))
    n2 = unifint(diff_lb, diff_ub, (1, h * w))
    n3 = unifint(diff_lb, diff_ub, (1, h * w))
    n4 = unifint(diff_lb, diff_ub, (1, h * w))
    i1 = sample(inds, n1)
    i2 = sample(inds, n2)
    i3 = sample(inds, n3)
    i4 = sample(inds, n4)
    gi = canvas(bgc, (2 * h + 4, 2 * w + 4))
    gi = fill(gi, barc, connect((1, 0), (1, fullw - 1)))
    gi = fill(gi, barc, connect((fullh - 2, 0), (fullh - 2, fullw - 1)))
    gi = fill(gi, barc, connect((0, 1), (fullh - 1, 1)))
    gi = fill(gi, barc, connect((0, fullw - 2), (fullh - 1, fullw - 2)))
    gi = fill(gi, c1, {(0, 0)})
    gi = fill(gi, c2, {(0, fullw - 1)})
    gi = fill(gi, c3, {(fullh - 1, 0)})
    gi = fill(gi, c4, {(fullh - 1, fullw - 1)})
    gi = fill(gi, inc, shift(i1, (2, 2)))
    gi = fill(gi, inc, shift(i2, (2, 2+w)))
    gi = fill(gi, inc, shift(i3, (2+h, 2)))
    gi = fill(gi, inc, shift(i4, (2+h, 2+w)))
    go = canvas(bgc, (2 * h, 2 * w))
    go = fill(go, c1, shift(i1, (0, 0)))
    go = fill(go, c2, shift(i2, (0, w)))
    go = fill(go, c3, shift(i3, (h, 0)))
    go = fill(go, c4, shift(i4, (h, w)))
    return {'input': gi, 'output': go}


def generate_d406998b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    itv = interval(0, h, 1)
    for j in range(w):
        nilocs = unifint(diff_lb, diff_ub, (1, h // 2 - 1 if h % 2 == 0 else h // 2))
        ilocs = sample(itv, nilocs)
        locs = {(ii, j) for ii in ilocs}
        gi = fill(gi, dotc, locs)
        go = fill(go, dotc if (j - w) % 2 == 0 else 3, locs)
    return {'input': gi, 'output': go}


def generate_694f12f3(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    seploc = randint(4, h - 5)
    bigh = unifint(diff_lb, diff_ub, (4, seploc))
    bigw = unifint(diff_lb, diff_ub, (3, w - 1))
    bigloci = randint(0, seploc - bigh)
    biglocj = randint(0, w - bigw)
    smallmaxh = h - seploc - 1
    smallmaxw = w - 1
    cands = []
    bigsize = bigh * bigw
    for a in range(3, smallmaxh+1):
        for b in range(3, smallmaxw+1):
            if a * b < bigsize:
                cands.append((a, b))
    cands = sorted(cands, key=lambda ab: ab[0]*ab[1])
    num = len(cands)
    idx = unifint(diff_lb, diff_ub, (0, num - 1))
    smallh, smallw = cands[idx]
    smallloci = randint(seploc+1, h - smallh)
    smalllocj = randint(0, w - smallw)
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    bigsq = backdrop(frozenset({(bigloci, biglocj), (bigloci + bigh - 1, biglocj + bigw - 1)}))
    smallsq = backdrop(frozenset({(smallloci, smalllocj), (smallloci + smallh - 1, smalllocj + smallw - 1)}))
    gi = fill(gi, sqc, bigsq | smallsq)
    go = fill(gi, 2, backdrop(inbox(bigsq)))
    go = fill(go, 1, backdrop(inbox(smallsq)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_3befdf3e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, ((h * w) // 40)))
    succ = 0
    maxtr = 5 * nobjs
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        rh = choice((1, 2))
        rw = choice((1, 2))
        fullh = (2 + 3 * rh)
        fullw = (2 + 3 * rw)
        cands = sfilter(inds, lambda ij: ij[0] <= h - fullh and ij[1] <= w - fullw)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        fullobj = backdrop(frozenset({loc, (loci + fullh - 1, locj + fullw - 1)}))
        if fullobj.issubset(inds):
            succ += 1
            inds = inds - fullobj
            incol, outcol = sample(ccols, 2)
            ofincol = backdrop(frozenset({(loci + rh + 1, locj + rw + 1), (loci + 2 * rh, locj + 2 * rw)}))
            ofoutcol = outbox(ofincol)
            gi = fill(gi, incol, ofincol)
            gi = fill(gi, outcol, ofoutcol)
            go = fill(go, outcol, ofincol)
            go = fill(go, incol, ofoutcol)
            ilocs = apply(first, ofoutcol)
            jlocs = apply(last, ofoutcol)
            ff = lambda ij: ij[0] in ilocs or ij[1] in jlocs
            addon = sfilter(fullobj - (ofincol | ofoutcol), ff)
            go = fill(go, outcol, addon)
    return {'input': gi, 'output': go}


def generate_9f236235(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    numh = unifint(diff_lb, diff_ub, (2, 14))
    numw = unifint(diff_lb, diff_ub, (2, 14))
    h = unifint(diff_lb, diff_ub, (1, 31 // numh - 1))
    w = unifint(diff_lb, diff_ub, (1, 31 // numw - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    frontcol = choice(remcols)
    remcols = remove(frontcol, cols)
    numcols = unifint(diff_lb, diff_ub, (1, min(9, numh * numw)))
    ccols = sample(remcols, numcols)
    numcells = unifint(diff_lb, diff_ub, (1, numh * numw))
    cands = asindices(canvas(-1, (numh, numw)))
    inds = asindices(canvas(-1, (h, w)))
    locs = sample(totuple(cands), numcells)
    gi = canvas(frontcol, (h * numh + numh - 1, w * numw + numw - 1))
    go = canvas(bgc, (numh, numw))
    for cand in cands:
        a, b = cand
        plcd = shift(inds, (a * (h + 1), b * (w + 1)))
        col = choice(remcols) if cand in locs else bgc
        gi = fill(gi, col, plcd)
        go = fill(go, col, {cand})
    go = vmirror(go)
    return {'input': gi, 'output': go}


def generate_d8c310e9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    p = unifint(diff_lb, diff_ub, (2, (w - 1) // 3))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        numcells = unifint(diff_lb, diff_ub, (1, h - 1))
        for ii in range(h - 1, h - numcells - 1, -1):
            loc = (ii, j)
            col = choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    gi = canvas(bgc, (h, w))
    minobj = obj | shift(obj, (0, p))
    addonw = randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    gi = paint(gi, fullobj)
    go = tuple(e for e in gi)
    for j in range(w//(2*p)+2):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    mfs = (identity, rot90, rot180, rot270)
    fn = choice(mfs)
    gi = fn(gi)
    go = fn(go)
    return {'input': gi, 'output': go}


def generate_7e0986d6(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nsqcols = unifint(diff_lb, diff_ub, (1, 5))
    sqcols = sample(remcols, nsqcols)
    remcols = difference(remcols, sqcols)
    nnoisecols = unifint(diff_lb, diff_ub, (1, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * numsq
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - outbox(sq)
            col = choice(sqcols)
            go = fill(go, col, sq)
    gi = tuple(e for e in go)
    namt = unifint(diff_lb, diff_ub, (1, (h * w) // 9))
    cands = asindices(gi)
    for k in range(namt):
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        col = gi[loc[0]][loc[1]]
        torem = neighbors(loc) & ofcolor(gi, col)
        cands = cands - torem
        noisec = choice(noisecols)
        gi = fill(gi, noisec, {loc})
    return {'input': gi, 'output': go}


def generate_a64e4611(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (18, 30))
    w = unifint(diff_lb, diff_ub, (18, 30))
    bgc, noisec = sample(cols, 2)
    lb = int(0.4 * h * w)
    ub = int(0.5 * h * w)
    nbgc = unifint(diff_lb, diff_ub, (lb, ub))
    gi = canvas(noisec, (h, w))
    inds = totuple(asindices(gi))
    bgcinds = sample(inds, nbgc)
    gi = fill(gi, bgc, bgcinds)
    sinds = asindices(canvas(-1, (3, 3)))
    bgcf = recolor(bgc, sinds)
    noisecf = recolor(noisec, sinds)
    addn = set()
    addb = set()
    for occ in occurrences(gi, bgcf):
        occi, occj = occ
        addn.add((randint(0, 2) + occi, randint(0, 2) + occj))
    for occ in occurrences(gi, noisecf):
        occi, occj = occ
        addb.add((randint(0, 2) + occi, randint(0, 2) + occj))
    gi = fill(gi, noisec, addn)
    gi = fill(gi, bgc, addb)
    go = tuple(e for e in gi)
    dim = randint(randint(3, 8), 8)
    locj = randint(3, h - dim - 4)
    spi = choice((0, randint(3, h//2)))
    for j in range(locj, locj + dim):
        ln = connect((spi, j), (h - 1, j))
        gi = fill(gi, bgc, ln)
        go = fill(go, bgc, ln)
    for j in range(locj + 1, locj + dim - 1):
        ln = connect((spi + 1 if spi > 0 else spi, j), (h - 1, j))
        go = fill(go, 3, ln)
    sgns = choice(((-1,), (1,), (-1, 1)))
    startloc = choice((spi, randint(spi + 3, h - 6)))
    hh = randint(3, min(8, h - startloc - 3))
    for sgn in sgns:
        for ii in range(startloc, startloc + hh, 1):
            ln = shoot((ii, locj), (0, sgn))
            gi = fill(gi, bgc, ln)
            go = fill(go, bgc, ln - ofcolor(go, 3))
    for sgn in sgns:
        for ii in range(startloc+1 if startloc > 0 else startloc, startloc + hh - 1, 1):
            ln = shoot((ii, locj+dim-2 if sgn == -1 else locj+1), (0, sgn))
            go = fill(go, 3, ln)
    if len(sgns) == 1 and unifint(diff_lb, diff_ub, (0, 1)) == 1:
        sgns = (-sgns[0],)
        startloc = choice((spi, randint(spi + 3, h - 6)))
        hh = randint(3, min(8, h - startloc - 3))
        for sgn in sgns:
            for ii in range(startloc, startloc + hh, 1):
                ln = shoot((ii, locj), (0, sgn))
                gi = fill(gi, bgc, ln)
                go = fill(go, bgc, ln - ofcolor(go, 3))
        for sgn in sgns:
            for ii in range(startloc+1 if startloc > 0 else startloc, startloc + hh - 1, 1):
                ln = shoot((ii, locj+dim-2 if sgn == -1 else locj+1), (0, sgn))
                go = fill(go, 3, ln)
    return {'input': gi, 'output': go}


def generate_b782dc8a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}
    dlt = [('W', (-1, 0)), ('E', (1, 0)), ('S', (0, 1)), ('N', (0, -1))]
    walls = {'N': True, 'S': True, 'E': True, 'W': True}
    fullsucc = False
    while True:
        h = unifint(diff_lb, diff_ub, (3, 15))
        w = unifint(diff_lb, diff_ub, (3, 15))
        maze = [[{'x': x, 'y': y, 'walls': {**walls}} for y in range(h)] for x in range(w)]
        kk = h * w
        stck = []
        cc = maze[0][0]
        nv = 1
        while nv < kk:
            nbhs = []
            for direc, (dx, dy) in dlt:
                x2, y2 = cc['x'] + dx, cc['y'] + dy
                if 0 <= x2 < w and 0 <= y2 < h:
                    neighbour = maze[x2][y2]
                    if all(neighbour['walls'].values()):
                        nbhs.append((direc, neighbour))
            if not nbhs:
                cc = stck.pop()
                continue
            direc, next_cell = choice(nbhs)
            cc['walls'][direc] = False
            next_cell['walls'][wall_pairs[direc]] = False
            stck.append(cc)
            cc = next_cell
            nv += 1
        pathcol, wallcol, dotcol, ncol = sample(cols, 4)
        grid = [[pathcol for x in range(w * 2)]]
        for y in range(h):
            row = [pathcol]
            for x in range(w):
                row.append(wallcol)
                row.append(pathcol if maze[x][y]['walls']['E'] else wallcol)
            grid.append(row)
            row = [pathcol]
            for x in range(w):
                row.append(pathcol if maze[x][y]['walls']['S'] else wallcol)
                row.append(pathcol)
            grid.append(row)
        gi = tuple(tuple(r[1:-1]) for r in grid[1:-1])
        objs = objects(gi, T, F, F)
        objs = colorfilter(objs, pathcol)
        objs = sfilter(objs, lambda obj: size(obj) > 4)
        if len(objs) == 0:
            continue
        objs = order(objs, size)
        nobjs = len(objs)
        idx = unifint(diff_lb, diff_ub, (0, nobjs - 1))
        obj = toindices(objs[idx])
        cell = choice(totuple(obj))
        gi = fill(gi, dotcol, {cell})
        nbhs = dneighbors(cell) & ofcolor(gi, pathcol)
        gi = fill(gi, ncol, nbhs)
        obj1 = sfilter(obj, lambda ij: even(manhattan({ij}, {cell})))
        obj2 = obj - obj1
        go = fill(gi, dotcol, obj1)
        go = fill(go, ncol, obj2)
        break
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_af902bf9(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    maxtr = 5 * numsq
    tr = 0
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < numsq:
        tr += 1
        oh = randint(3, 5)
        ow = randint(3, 5)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            inds = inds - sq
            succ += 1
            col = choice(ccols)
            crns = corners(sq)
            gi = fill(gi, col, crns)
            go = fill(go, col, crns)
            ins = backdrop(inbox(crns))
            go = fill(go, 2, ins)
    return {'input': gi, 'output': go}


def generate_a87f7484(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 30))
    num = unifint(diff_lb, diff_ub, (3, min(30 // h, 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, num)
    ncd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc = choice((ncd, h * w - ncd))
    nc = min(max(1, nc), h * w - 1)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    origlocs = sample(totuple(inds), nc)
    canbrem = {l for l in origlocs}
    canbeadd = inds - set(origlocs)
    otherlocs = {l for l in origlocs}
    nchangesinv = unifint(diff_lb, diff_ub, (0, h * w - 1))
    nchanges = h * w - nchangesinv
    for k in range(nchanges):
        if choice((True, False)):
            if len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
            elif len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
        else:
            if len(canbeadd) > 1:
                ch = choice(totuple(canbeadd))
                otherlocs = insert(ch, otherlocs)
                canbeadd = remove(ch, canbeadd)
            elif len(canbrem) > 1:
                ch = choice(totuple(canbrem))
                otherlocs = remove(ch, otherlocs)
                canbrem = remove(ch, canbrem)
    go = fill(c, ccols[0], origlocs)
    grids = [go]
    for cc in ccols[1:]:
        grids.append(fill(c, cc, otherlocs))
    shuffle(grids)
    grids = tuple(grids)
    gi = merge(grids)
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_fcc82909(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, w // 3))
    opts = interval(0, w, 1)
    tr = 0
    maxtr = 4 * nobjs
    succ = 0
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    while succ < nobjs and tr < maxtr:
        tr += 1
        sopts = sfilter(opts, lambda j: set(interval(j, j + 2, 1)).issubset(opts))
        if len(sopts) == 0:
            break
        numc = unifint(diff_lb, diff_ub, (1, 4))
        jstart = choice(sopts)
        opts = remove(jstart, opts)
        opts = remove(jstart+1, opts)
        options = interval(0, h - 2 - numc + 1, 1)
        if len(options) == 0:
            break
        iloc = choice(options)
        ccols = sample(remcols, numc)
        bd = backdrop(frozenset({(iloc, jstart), (iloc + 1, jstart + 1)}))
        bd = list(bd)
        shuffle(bd)
        obj = {(c, ij) for c, ij in zip(ccols, bd[:numc])} | {(choice(ccols), ij) for ij in bd[numc:]}
        if not mapply(dneighbors, toindices(obj)).issubset(ofcolor(gi, bgc)):
            continue
        gi = paint(gi, obj)
        go = paint(go, obj)
        for k in range(numc):
            go = fill(go, 3, {(iloc+k+2, jstart), (iloc+k+2, jstart+1)})
    return {'input': gi, 'output': go}


def generate_d9fac9be(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, noisec, ringc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    nnoise1 = unifint(diff_lb, diff_ub, (1, (h * w) // 3 - 1))
    nnoise2 = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 3 - 9)))
    inds = asindices(gi)
    noise1 = sample(totuple(inds), nnoise1)
    noise2 = sample(difference(totuple(inds), noise1), nnoise2)
    gi = fill(gi, noisec, noise1)
    gi = fill(gi, ringc, noise2)
    rng = neighbors((1, 1))
    fp1 = recolor(noisec, rng)
    fp2 = recolor(ringc, rng)
    fp1occ = occurrences(gi, fp1)
    fp2occ = occurrences(gi, fp2)
    for occ1 in fp1occ:
        loc = choice(totuple(shift(rng, occ1)))
        gi = fill(gi, choice((bgc, ringc)), {loc})
    for occ2 in fp2occ:
        loc = choice(totuple(shift(rng, occ2)))
        gi = fill(gi, choice((bgc, noisec)), {loc})
    loci = randint(0, h - 3)
    locj = randint(0, w - 3)
    ringp = shift(rng, (loci, locj))
    gi = fill(gi, ringc, ringp)
    gi = fill(gi, noisec, {(loci + 1, locj + 1)})
    go = canvas(noisec, (1, 1))
    return {'input': gi, 'output': go}


def generate_eb281b96(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 8))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    ncells = unifint(diff_lb, diff_ub, (1, h * w))
    locs = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in locs}
    gi = paint(c, obj)
    go = vconcat(gi, hmirror(gi[:-1]))
    go = vconcat(go, hmirror(go[:-1]))
    return {'input': gi, 'output': go}


def generate_d43fd935(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    boxh = unifint(diff_lb, diff_ub, (2, h // 2))
    boxw = unifint(diff_lb, diff_ub, (2, w // 2))
    loci = randint(0, h - boxh)
    locj = randint(0, w - boxw)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccol = choice(remcols)
    remcols = remove(ccol, remcols)
    ndcols = unifint(diff_lb, diff_ub, (1, 8))
    dcols = sample(remcols, ndcols)
    bd = backdrop(frozenset({(loci, locj), (loci + boxh - 1, locj + boxw - 1)}))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, ccol, bd)
    reminds = totuple(asindices(gi) - bd)
    noiseb = max(1, len(reminds) // 4)
    nnoise = unifint(diff_lb, diff_ub, (0, noiseb))
    noise = sample(reminds, nnoise)
    truenoise = sfilter(noise, lambda ij: (ij[0] < loci or ij[0] > loci + boxh - 1) and (ij[1] < locj or ij[1] > locj + boxw - 1))
    rem = difference(noise, truenoise)
    top = sfilter(rem, lambda ij: ij[0] < loci)
    bottom = sfilter(rem, lambda ij: ij[0] > loci + boxh - 1)
    left = sfilter(rem, lambda ij: ij[1] < locj)
    right = sfilter(rem, lambda ij: ij[1] > locj + boxw - 1)
    truenoiseobj = {(choice(dcols), ij) for ij in truenoise}
    gi = paint(gi, truenoiseobj)
    go = tuple(e for e in gi)
    for jj in apply(last, top):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(top, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmin(subs, first), jj), (loci - 1, jj)))
    for jj in apply(last, bottom):
        col = choice(dcols)
        mf = matcher(last, jj)
        subs = sfilter(bottom, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((valmax(subs, first), jj), (loci + boxh, jj)))
    for ii in apply(first, left):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(left, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmin(subs, last)), (ii, locj - 1)))
    for ii in apply(first, right):
        col = choice(dcols)
        mf = matcher(first, ii)
        subs = sfilter(right, mf)
        gi = fill(gi, col, subs)
        go = fill(go, col, connect((ii, valmax(subs, last)), (ii, locj + boxw)))
    return {'input': gi, 'output': go}


def generate_44f52bb0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    numcells = unifint(diff_lb, diff_ub, (1, h * w - 1))
    inds = asindices(gi)
    while gi == hmirror(gi):
        cells = sample(totuple(inds), numcells)
        gi = canvas(bgc, (h, w))
        for ij in cells:
            a, b = ij
            col = choice(ccols)
            gi = fill(gi, col, {ij})
            gi = fill(gi, col, {(a, w - 1 - b)})
    issymm = choice((True, False))
    if not issymm:
        numpert = unifint(diff_lb, diff_ub, (1, h * (w // 2)))
        cands = asindices(canvas(-1, (h, w // 2)))
        locs = sample(totuple(cands), numpert)
        for a, b in locs:
            col = gi[a][b]
            newcol = choice(totuple(remove(col, insert(bgc, set(ccols)))))
            gi = fill(gi, newcol, {(a, b)})
        go = canvas(7, (1, 1))
    else:
        go = canvas(1, (1, 1))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_d22278a0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)    
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    crns = corners(inds)
    ncorns = unifint(diff_lb, diff_ub, (1, 4))
    crns = sample(totuple(crns), ncorns)
    ccols = sample(remcols, ncorns)
    for col, crn in zip(ccols, crns):
        gi = fill(gi, col, {crn})
        go = fill(go, col, {crn})
        rings = {crn}
        for k in range(1, max(h, w) // 2 + 2, 1):
            rings = rings | outbox(outbox(rings))
        if len(crns) > 1:
            ff = lambda ij: manhattan({ij}, {crn}) < min(apply(rbind(manhattan, {ij}), apply(initset, remove(crn, crns))))
        else:
            ff = lambda ij: True
        locs = sfilter(inds, ff) & rings
        go = fill(go, col, locs)
    return {'input': gi, 'output': go}


def generate_272f95fa(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 4, 6))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (5, 5))
    l1 = connect((1, 0), (1, 4))
    l2 = connect((3, 0), (3, 4))
    lns = l1 | l2
    gi = fill(dmirror(fill(c, linc, lns)), linc, lns)
    hdist = [0, 0, 0]
    wdist = [0, 0, 0]
    idx = 0
    for k in range(h - 2):
        hdist[idx] += 1
        idx = (idx + 1) % 3
    for k in range(w - 2):
        wdist[idx] += 1
        idx = (idx + 1) % 3
    shuffle(hdist)
    shuffle(wdist)
    hdelt1 = unifint(diff_lb, diff_ub, (0, hdist[0] - 1))
    hdist[0] -= hdelt1
    hdist[1] += hdelt1
    hdelt2 = unifint(diff_lb, diff_ub, (0, min(hdist[1], hdist[2]) - 1))
    hdelt2 = choice((+hdelt2, -hdelt2))
    hdist[1] += hdelt2
    hdist[2] -= hdelt2
    wdelt1 = unifint(diff_lb, diff_ub, (0, wdist[0] - 1))
    wdist[0] -= wdelt1
    wdist[1] += wdelt1
    wdelt2 = unifint(diff_lb, diff_ub, (0, min(wdist[1], wdist[2]) - 1))
    wdelt2 = choice((+wdelt2, -wdelt2))
    wdist[1] += wdelt2
    wdist[2] -= wdelt2
    gi = gi[:1] * hdist[0] + gi[1:2] + gi[2:3] * hdist[1] + gi[3:4] + gi[4:5] * hdist[2]
    gi = dmirror(gi)
    gi = gi[:1] * wdist[0] + gi[1:2] + gi[2:3] * wdist[1] + gi[3:4] + gi[4:5] * wdist[2]
    gi = dmirror(gi)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
    objs = objects(gi, T, T, F)
    bgobjs = colorfilter(objs, bgc)
    cnrs = corners(asindices(gi))
    bgobjs = sfilter(bgobjs, lambda o: len(toindices(o) & cnrs) == 0)
    pinkobj = extract(bgobjs, lambda o: not bordering(o, gi))
    yellobj = argmin(bgobjs, leftmost)
    greenobj = argmax(bgobjs, rightmost)
    redobj = argmin(bgobjs, uppermost)
    blueobj = argmax(bgobjs, lowermost)
    go = fill(gi, 6, pinkobj)
    go = fill(go, 4, yellobj)
    go = fill(go, 3, greenobj)
    go = fill(go, 2, redobj)
    go = fill(go, 1, blueobj)
    return {'input': gi, 'output': go}


def generate_5c0a986e(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))    
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    inds = asindices(gi)
    fullinds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: 0 < ij[0] <= h - 3 and 0 < ij[1] <= w - 3)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        col = choice((1, 2))
        sq = {(loc), add(loc, (0, 1)), add(loc, (1, 0)), add(loc, (1, 1))}
        if col == 1:
            obj = sq | (shoot(loc, (-1, -1)) & fullinds)
        else:
            obj = sq | (shoot(loc, (1, 1)) & fullinds)
        if obj.issubset(inds):
            succ += 1
            inds = (inds - obj) - mapply(dneighbors, sq)
            gi = fill(gi, col, sq)
            go = fill(go, col, obj)
    return {'input': gi, 'output': go}


def generate_9af7a82c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    prods = dict()
    for a in range(1, 31, 1):
        for b in range(1, 31, 1):
            prd = a*b
            if prd in prods:
                prods[prd].append((a, b))
            else:
                prods[prd] = [(a, b)]
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    leastnc = sum(range(1, ncols + 1, 1))
    maxnc = sum(range(30, 30 - ncols, -1))
    cands = {k: v for k, v in prods.items() if leastnc <= k <= maxnc}
    options = set()
    for v in cands.values():
        for opt in v:
            options.add(opt)
    options = sorted(options, key=lambda ij: ij[0] * ij[1])
    idx = unifint(diff_lb, diff_ub, (0, len(options) - 1))
    h, w = options[idx]
    ccols = sample(cols, ncols)
    counts = list(range(1, ncols + 1, 1))
    eliginds = {ncols - 1}
    while sum(counts) < h * w:
        eligindss = sorted(eliginds, reverse=True)
        idx = unifint(diff_lb, diff_ub, (0, len(eligindss) - 1))
        idx = eligindss[idx]
        counts[idx] += 1
        if idx > 0:
            eliginds.add(idx - 1)
        if idx < ncols - 1:
            if counts[idx] == counts[idx+1] - 1:
                eliginds = eliginds - {idx}
        if counts[idx] == 30:
            eliginds = eliginds - {idx}
    gi = canvas(-1, (h, w))
    go = canvas(0, (max(counts), ncols))
    inds = asindices(gi)
    counts = counts[::-1]
    for j, (col, cnt) in enumerate(zip(ccols, counts)):
        locs = sample(totuple(inds), cnt)
        gi = fill(gi, col, locs)
        inds = inds - set(locs)
        go = fill(go, col, connect((0, j), (cnt - 1, j)))
    return {'input': gi, 'output': go}


def generate_d4469b4b(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3))
    canv = canvas(5, (3, 3))
    A = fill(canv, 0, {(1, 0), (2, 0), (1, 2), (2, 2)})
    B = fill(canv, 0, corners(asindices(canv)))
    C = fill(canv, 0, {(0, 0), (0, 1), (1, 0), (1, 1)})
    colabc = ((2, A), (1, B), (3, C))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    col, go = choice(colabc)
    gi = canvas(col, (h, w))
    inds = asindices(gi)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(cols, numc)
    numcells = unifint(diff_lb, diff_ub, (0, h * w - 1))
    locs = sample(totuple(inds), numcells)
    otherobj = {(choice(ccols), ij) for ij in locs}
    gi = paint(gi, otherobj)
    return {'input': gi, 'output': go}


def generate_bdad9b1f(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    numh = unifint(diff_lb, diff_ub, (1, h // 2 - 1))
    numw = unifint(diff_lb, diff_ub, (1, w // 2 - 1))
    hlocs = sample(interval(2, h - 1, 1), numh)
    wlocs = sample(interval(2, w - 1, 1), numw)
    numcols = unifint(diff_lb, diff_ub, (2, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, numcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    fc = -1
    for ii in sorted(hlocs):
        col = choice(remove(fc, ccols))
        fc = col
        objw = randint(2, ii)
        gi = fill(gi, col, connect((ii, 0), (ii, objw - 1)))
        go = fill(go, col, connect((ii, 0), (ii, w - 1)))
    fc = -1
    for jj in sorted(wlocs):
        col = choice(remove(fc, ccols))
        fc = col
        objh = randint(2, jj)
        gi = fill(gi, col, connect((0, jj), (objh - 1, jj)))
        go = fill(go, col, connect((0, jj), (h - 1, jj)))
    yells = product(set(hlocs), set(wlocs))
    go = fill(go, 4, yells)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_3345333e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (4, h - 2))
    ow = unifint(diff_lb, diff_ub, (4, (w - 2) // 2))
    nc = unifint(diff_lb, diff_ub, (min(oh, ow), (oh * ow) // 3 * 2))
    shp = {(0, 0)}
    bounds = asindices(canvas(-1, (oh, ow)))
    for j in range(nc):
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    while height(shp) < 3 or width(shp) < 3:
        ij = choice(totuple((bounds - shp) & mapply(neighbors, shp)))
        shp.add(ij)
    vmshp = vmirror(shp)
    if choice((True, False)):
        vmshp = sfilter(vmshp, lambda ij: ij[1] != width(shp) - 1)
    shp = normalize(combine(shp, shift(vmshp, (0, -width(vmshp)))))
    oh, ow = shape(shp)
    bgc, objc, occcol = sample(cols, 3)
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    loc = (loci, locj)
    shp = shift(shp, loc)
    c = canvas(bgc, (h, w))
    go = fill(c, objc, shp)
    boxh = unifint(diff_lb, diff_ub, (2, oh - 1))
    boxw = unifint(diff_lb, diff_ub, (2, ow//2))
    ulci = randint(loci - 1, loci + oh - boxh + 1)
    ulcj = randint(locj + ow//2 + 1, locj + ow - boxw + 1)
    bx = backdrop(frozenset({(ulci, ulcj), (ulci + boxh - 1, ulcj + boxw - 1)}))
    gi = fill(go, occcol, bx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_253bf280(diff_lb: float, diff_ub: float) -> dict:
    dim_bounds = (3, 30)
    colopts = remove(3, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, dim_bounds)
    w = unifint(diff_lb, diff_ub, dim_bounds)
    bgc = choice(colopts)
    c = canvas(bgc, (h, w))
    inds = totuple(asindices(c))
    card_bounds = (0, max(1, (h * w) // 4))
    num = unifint(diff_lb, diff_ub, card_bounds)
    s = sample(inds, num)
    fgcol = choice(remove(bgc, colopts))
    gi = fill(c, fgcol, s)
    resh = frozenset()
    for x, r in enumerate(gi):
        if r.count(fgcol) > 1:
            resh = combine(resh, connect((x, r.index(fgcol)), (x, -1 + w - r[::-1].index(fgcol))))
    go = fill(c, 3, resh)
    resv = frozenset()
    for x, r in enumerate(dmirror(gi)):
        if r.count(fgcol) > 1:
            resv = combine(resv, connect((x, r.index(fgcol)), (x, -1 + h - r[::-1].index(fgcol))))
    go = dmirror(fill(dmirror(go), 3, resv))
    go = fill(go, fgcol, s)
    return {'input': gi, 'output': go}


def generate_5582e5ca(diff_lb: float, diff_ub: float) -> dict:
    colopts = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    numc = unifint(diff_lb, diff_ub, (2, min(10, h * w - 1)))
    ccols = sample(colopts, numc)
    mostc = ccols[0]
    remcols = ccols[1:]
    leastnummostcol = (h * w) // numc + 1
    maxnummostcol = h * w - numc + 1
    nummostcold = unifint(diff_lb, diff_ub, (0, maxnummostcol - leastnummostcol))
    nummostcol = min(max(leastnummostcol, maxnummostcol - nummostcold), maxnummostcol)
    kk = len(remcols)
    remcount = h * w - nummostcol - kk
    remcounts = [1 for k in range(kk)]
    for j in range(remcount):
        cands = [idx for idx, c in enumerate(remcounts) if c < nummostcol - 1]
        if len(cands) == 0:
            break
        idx = choice(cands)
        remcounts[idx] += 1
    nummostcol = h * w - sum(remcounts)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    mclocs = sample(totuple(inds), nummostcol)
    gi = fill(gi, mostc, mclocs)
    go = canvas(mostc, (h, w))
    inds = inds - set(mclocs)
    for col, count in zip(remcols, remcounts):
        locs = sample(totuple(inds), count)
        inds = inds - set(locs)
        gi = fill(gi, col, locs)
    return {'input': gi, 'output': go}


def generate_a1570a43(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    oh = unifint(diff_lb, diff_ub, (3, h))
    ow = unifint(diff_lb, diff_ub, (3, w))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    crns = {(loci, locj), (loci + oh - 1, locj), (loci, locj + ow - 1), (loci + oh - 1, locj + ow - 1)}
    cands = shift(asindices(canvas(-1, (oh-2, ow-2))), (loci+1, locj+1))
    bgc, dotc = sample(cols, 2)
    remcols = remove(bgc, remove(dotc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    gipro = canvas(bgc, (h, w))
    gipro = fill(gipro, dotc, crns)
    sp = choice(totuple(cands))
    obj = {sp}
    cands = remove(sp, cands)
    ncells = unifint(diff_lb, diff_ub, (oh + ow - 5, max(oh + ow - 5, ((oh - 2) * (ow - 2)) // 2)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    while shape(obj) != (oh-2, ow-2):
        obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
    obj = {(choice(ccols), ij) for ij in obj}
    go = paint(gipro, obj)
    nperts = unifint(diff_lb, diff_ub, (1, max(h, w)))
    k = 0
    fullinds = asindices(go)
    while ulcorner(obj) == (loci+1, locj+1) or k < nperts:
        k += 1
        options = sfilter(
            neighbors((0, 0)),
            lambda ij: len(crns & shift(toindices(obj), ij)) == 0 and \
                shift(toindices(obj), ij).issubset(fullinds)
        )
        direc = choice(totuple(options))
        obj = shift(obj, direc)
    gi = paint(gipro, obj)
    return {'input': gi, 'output': go}


def generate_f5b8619d(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, (h * w) // 2 - 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    locs = sample(totuple(inds), ncells)
    blockcol = randint(0, w - 1)
    locs = sfilter(locs, lambda ij: ij[1] != blockcol)
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(gi, obj)
    go = fill(gi, 8, mapply(vfrontier, set(locs)) & (inds - set(locs)))
    go = hconcat(go, go)
    go = vconcat(go, go)
    return {'input': gi, 'output': go}


def generate_444801d8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numcols)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(4, 6)
        ow = 5
        bx = box({(1, 0), (oh - 1, 4)}) - {(1, 2)}
        fullobj = backdrop({(0, 0), (oh - 1, 4)})
        cands = backdrop(bx) - bx
        dot = choice(totuple(cands))
        dcol, bxcol = sample(ccols, 2)
        inobj = recolor(bxcol, bx) | recolor(dcol, {dot})
        outobj = recolor(bxcol, bx) | recolor(dcol, fullobj - bx)
        if choice((True, False)):
            inobj = shift(hmirror(inobj), UP)
            outobj = hmirror(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        outplcd = shift(outobj, loc)
        outplcdi = toindices(outplcd)
        if outplcdi.issubset(inds):
            succ += 1
            inplcd = shift(inobj, loc)
            inds = (inds - outplcdi) - outbox(inplcd)
            gi = paint(gi, inplcd)
            go = paint(go, outplcd)
    return {'input': gi, 'output': go}


def generate_00d62c1b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    nblocks = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nblocks
    inds = asindices(gi)
    while succ < nblocks and tr < maxtr:
        tr += 1
        oh = randint(3, 8)
        ow = randint(3, 8)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bx = bx - set(sample(totuple(corners(bx)), randint(0, 4)))
        if bx.issubset(inds) and len(inds - bx) > (h * w) // 2 + 1:
            gi = fill(gi, fgc, bx)
            succ += 1
            inds = inds - bx
    maxnnoise = max(0, (h * w) // 2 - 1 - colorcount(gi, fgc))
    namt = unifint(diff_lb, diff_ub, (0, maxnnoise))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    objs = objects(gi, T, F, F)
    cands = colorfilter(objs, bgc)
    res = mfilter(cands, compose(flip, rbind(bordering, gi)))
    go = fill(gi, 4, res)
    return {'input': gi, 'output': go}


def generate_10fcaaa3(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (2, 15))
    w = unifint(diff_lb, diff_ub, (2, 15))
    ncells = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 6)))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, ncols)
    c = canvas(bgc, (h, w))
    inds = asindices(c)
    locs = frozenset(sample(totuple(inds), ncells))
    obj = frozenset({(choice(ccols), ij) for ij in locs})
    gi = paint(c, obj)
    go = hconcat(gi, gi)
    go = vconcat(go, go)
    fullocs = locs | shift(locs, (0, w)) | shift(locs, (h, 0)) | shift(locs, (h, w))
    nbhs = mapply(ineighbors, fullocs)
    topaint = nbhs & ofcolor(go, bgc)
    go = fill(go, 8, topaint)
    return {'input': gi, 'output': go}


def generate_1a07d186(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nlines = unifint(diff_lb, diff_ub, (1, w // 5))
    linecols = sample(remcols, nlines)
    remcols = difference(remcols, linecols)
    nnoisecols = unifint(diff_lb, diff_ub, (0, len(remcols)))
    noisecols = sample(remcols, nnoisecols)
    locopts = interval(0, w, 1)
    locs = []
    for k in range(nlines):
        if len(locopts) == 0:
            break
        loc = choice(locopts)
        locopts = difference(locopts, interval(loc - 2, loc + 3, 1))
        locs.append(loc)
    locs = sorted(locs)
    nlines = len(locs)
    linecols = linecols[:nlines]
    gi = canvas(bgc, (h, w))
    for loc, col in zip(locs, linecols):
        gi = fill(gi, col, connect((0, loc), (h - 1, loc)))
    go = tuple(e for e in gi)
    nilocs = unifint(diff_lb, diff_ub, (1, h))
    ilocs = sample(interval(0, h, 1), nilocs)
    dotlocopts = difference(interval(0, w, 1), locs)
    for ii in ilocs:
        ndots = unifint(diff_lb, diff_ub, (1, min(nlines + nnoisecols, (w - nlines) // 2 - 1)))
        dotlocs = sample(dotlocopts, ndots)
        dotcols = sample(totuple(set(linecols) | set(noisecols)), ndots)
        for dotlocj, col in zip(dotlocs, dotcols):
            gi = fill(gi, col, {(ii, dotlocj)})
            if col in linecols:
                idx = linecols.index(col)
                linelocj = locs[idx]
                if dotlocj > linelocj:
                    go = fill(go, col, {(ii, linelocj + 1)})
                else:
                    go = fill(go, col, {(ii, linelocj - 1)})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_83302e8f(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3, 4))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh = unifint(diff_lb, diff_ub, (3, 30 // (h + 1)))
    nw = unifint(diff_lb, diff_ub, (3, 30 // (w + 1)))
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h+1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w+1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    ofc = ofcolor(gi, linc)
    dots = sfilter(ofc, lambda ij: dneighbors(ij).issubset(ofc))
    tmp = fill(gi, bgc, dots)
    lns = apply(toindices, colorfilter(objects(tmp, T, F, F), linc))
    dts = apply(initset, dots)
    cands = lns | dts
    nbreaks = unifint(diff_lb, diff_ub, (0, len(cands) // 2))
    breaklocs = set()
    breakobjs = sample(totuple(cands), nbreaks)
    for breakobj in breakobjs:
        loc = choice(totuple(breakobj))
        breaklocs.add(loc)
    gi = fill(gi, bgc, breaklocs)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, bgc)
    objs = sfilter(objs, lambda o: len(o) == h * w)
    res = toindices(merge(objs))
    go = fill(gi, 3, res)
    go = replace(go, bgc, 4)
    return {'input': gi, 'output': go}


def generate_98cf29f8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    objh = unifint(diff_lb, diff_ub, (2, h - 5))
    objw = unifint(diff_lb, diff_ub, (2, w - 5))
    loci = randint(0, h - objh)
    locj = randint(0, w - objw)
    loc = (loci, locj)
    obj = backdrop(frozenset({(loci, locj), (loci + objh - 1, locj + objw - 1)}))
    bgc, objc, otherc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    bmarg = h - (loci + objh)
    rmarg = w - (locj + objw)
    tmarg = loci
    lmarg = locj
    margs = (bmarg, rmarg, tmarg, lmarg)
    options = [idx for idx, marg in enumerate(margs) if marg > 2]
    pos = choice(options)
    for k in range(pos):
        gi = rot90(gi)
    h, w = shape(gi)
    ofc = ofcolor(gi, objc)
    locis = randint(lowermost(ofc)+2, h-2)
    locie = randint(locis+1, h-1)
    locjs = randint(0, min(w - 2, rightmost(ofc)))
    locje = randint(max(locjs+1, leftmost(ofc)), w - 1)
    otherobj = backdrop(frozenset({(locis, locjs), (locie, locje)}))
    ub = min(rightmost(ofc), rightmost(otherobj))
    lb = max(leftmost(ofc), leftmost(otherobj))
    jloc = randint(lb, ub)
    ln = connect((lowermost(ofc)+1, jloc), (uppermost(otherobj)-1, jloc))
    gib = tuple(e for e in gi)
    gi = fill(gi, otherc, otherobj)
    gi = fill(gi, otherc, ln)
    go = fill(gib, otherc, shift(otherobj, (-len(ln), 0)))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_1f85a75f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(3, min(8, h // 2))
    ow = randint(3, min(8, w // 2))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = randint(max(oh, ow), oh * ow)
    sp = choice(totuple(bounds))
    obj = {sp}
    cands = remove(sp, bounds)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ((h * w) - len(backdrop(obj))) // 4)))
    gi = canvas(bgc, (h, w))
    obj = shift(obj, (loci, locj))
    gi = fill(gi, objc, obj)
    inds = asindices(gi)
    noisecells = sample(totuple(inds - backdrop(obj)), nnoise)
    noiseobj = frozenset({(choice(ccols), ij) for ij in noisecells})
    gi = paint(gi, noiseobj)
    go = fill(canvas(bgc, (oh, ow)), objc, normalize(obj))
    return {'input': gi, 'output': go}


def generate_8eb1be9a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    oh = unifint(diff_lb, diff_ub, (2, h // 3))
    ow = unifint(diff_lb, diff_ub, (2, w))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (2, (oh * ow) // 3 * 2))
    obj = normalize(frozenset(sample(totuple(bounds), ncells)))
    oh, ow = shape(obj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = frozenset({(choice(ccols), ij) for ij in obj})
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    obj = shift(obj, (loci, locj))
    c = canvas(bgc, (h, w))
    gi = paint(c, obj)
    go = paint(c, obj)
    for k in range(h // oh + 1):
        go = paint(go, shift(obj, (-oh*k, 0)))
        go = paint(go, shift(obj, (oh*k, 0)))
    return {'input': gi, 'output': go}


def generate_ba26e723(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 6))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w = unifint(diff_lb, diff_ub, (2, 30))
    gi = canvas(0, (h, w))
    go = canvas(0, (h, w))
    opts = interval(0, h, 1)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, ncols)
    for j in range(w):
        nc = unifint(diff_lb, diff_ub, (1, h - 1))
        locs = sample(opts, nc)
        obj = frozenset({(choice(ccols), (ii, j)) for ii in locs})
        gi = paint(gi, obj)
        if j % 3 == 0:
            obj = recolor(6, obj)
        go = paint(go, obj)
    return {'input': gi, 'output': go}


def generate_25d487eb(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    inds = asindices(go)
    while tr < maxtr and succ < nobjs:
        if len(inds) == 0:
            break
        tr += 1
        dim = randint(1, 3)
        obj = backdrop(frozenset({(0, 0), (dim, dim)}))
        obj = sfilter(obj, lambda ij: ij[0] <= ij[1])
        obj = obj | shift(vmirror(obj), (0, dim))
        mp = {(0, dim)}
        tric, linc = sample(ccols, 2)
        inobj = recolor(tric, obj - mp) | recolor(linc, mp)
        loc = choice(totuple(inds))
        iplcd = shift(inobj, loc)
        loci, locj = loc
        oplcd = iplcd | recolor(linc, connect((loci, locj + dim), (h - 1, locj + dim)) - toindices(iplcd))
        fullinds = asindices(gi)
        oplcdi = toindices(oplcd)
        if oplcdi.issubset(inds):
            succ += 1
            gi = paint(gi, iplcd)
            go = paint(go, oplcd)
        rotf = choice((identity, rot90, rot180, rot270))
        gi = rotf(gi)
        go = rotf(go)
        h, w = shape(gi)
        ofc = ofcolor(go, bgc)
        inds = ofc - mapply(dneighbors, asindices(go) - ofc)
    return {'input': gi, 'output': go}


def generate_4be741c5(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    numcolors = unifint(diff_lb, diff_ub, (2, w // 3))
    ccols = sample(cols, numcolors)
    go = (tuple(ccols),)
    gi = merge(tuple(repeat(repeat(c, h), 3) for c in ccols))
    while len(gi) < w:
        idx = randint(0, len(gi) - 1)
        gi = gi[:idx] + gi[idx:idx+1] + gi[idx:]
    gi = dmirror(gi)
    ndisturbances = unifint(diff_lb, diff_ub, (0, 3 * h * numcolors))
    for k in range(ndisturbances):
        options = []
        for a in range(h):
            for b in range(w - 3):
                if gi[a][b] == gi[a][b+1] and gi[a][b+2] == gi[a][b+3]:
                    options.append((a, b, gi[a][b], gi[a][b+2]))
        if len(options) == 0:
            break
        a, b, c1, c2 = choice(options)
        if choice((True, False)):
            gi = fill(gi, c2, {(a, b+1)})
        else:
            gi = fill(gi, c1, {(a, b+2)})
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_e509e548(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 6))
    getL = lambda h, w: connect((0, 0), (h - 1, 0)) | connect((0, 0), (0, w - 1))
    getU = lambda h, w: connect((0, 0), (0, w - 1)) | connect((0, 0), (randint(1, h - 1), 0)) | connect((0, w - 1), (randint(1, h - 1), w - 1))
    getH = lambda h, w: connect((0, 0), (0, w - 1)) | shift(connect((0, 0), (h - 1, 0)) | connect((h - 1, 0), (h - 1, randint(1, w - 1))), (0, randint(1, w - 2)))
    minshp_getter_pairs = ((2, 2, getL), (2, 3, getU), (3, 3, getH))
    colmapper = {getL: 1, getU: 6, getH: 2}
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 6))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        minh, minw, getter = choice(minshp_getter_pairs)
        oh = randint(minh, 6)
        ow = randint(minw, 6)
        obj = getter(oh, ow)
        mfs = (identity, dmirror, cmirror, vmirror, hmirror)
        nmfs = choice((1, 2))
        for fn in sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, colmapper[getter], plcd)
    return {'input': gi, 'output': go}


def generate_810b9b61(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3,))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 6))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (3, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(3, 5)
        ow = randint(3, 5)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        obj = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1  )}))
        mfs = (identity, dmirror, cmirror, vmirror, hmirror)
        nmfs = choice((1, 2))
        for fn in sample(mfs, nmfs):
            obj = fn(obj)
            obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if choice((True, False)):
            ninobjc = unifint(diff_lb, diff_ub, (1, len(plcd) - 1))
            inobj = frozenset(sample(totuple(plcd), ninobjc))
        else:
            inobj = plcd
        if inobj.issubset(inds):
            succ += 1
            inds = (inds - inobj) - mapply(dneighbors, inobj)
            col = choice(ccols)
            gi = fill(gi, col, inobj)
            go = fill(go, 3 if box(inobj) == inobj and min(shape(inobj)) > 2 else col, inobj)
    return {'input': gi, 'output': go}


def generate_6d0160f0(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (2, 5))
    w = unifint(diff_lb, diff_ub, (2, 5))
    nh, nw = h, w
    bgc, linc = sample(cols, 2)
    fullh = h * nh + nh - 1
    fullw = w * nw + nw - 1
    gi = canvas(bgc, (fullh, fullw))
    for iloc in range(h, fullh, h+1):
        gi = fill(gi, linc, hfrontier((iloc, 0)))
    for jloc in range(w, fullw, w+1):
        gi = fill(gi, linc, vfrontier((0, jloc)))
    noccs = unifint(diff_lb, diff_ub, (1, h * w))
    denseinds = asindices(canvas(-1, (h, w)))
    sparseinds = {(a*(h+1), b*(w+1)) for a, b in denseinds}
    locs = sample(totuple(sparseinds), noccs)
    trgtl = choice(locs)
    remlocs = remove(trgtl, locs)
    ntrgt = unifint(diff_lb, diff_ub, (1, (h * w - 1)))
    place = choice(totuple(denseinds))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    candss = totuple(remove(place, denseinds))
    trgrem = sample(candss, ntrgt)
    trgrem = {(choice(ccols), ij) for ij in trgrem}
    trgtobj = {(4, place)} | trgrem
    go = paint(gi, shift(sfilter(trgtobj, lambda cij: cij[0] != linc), multiply(place, increment((h, w)))))
    gi = paint(gi, shift(trgtobj, trgtl))
    toleaveout = ccols
    for rl in remlocs:
        tlo = choice(totuple(ccols))
        ncells = unifint(diff_lb, diff_ub, (1, h * w - 1))
        inds = sample(totuple(denseinds), ncells)
        obj = {(choice(remove(tlo, ccols) if len(ccols) > 1 else ccols), ij) for ij in inds}
        toleaveout = remove(tlo, toleaveout)
        gi = paint(gi, shift(obj, rl))
    return {'input': gi, 'output': go}


def generate_63613498(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sepc = sample(cols, 2)
    remcols = remove(bgc, remove(sepc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    objh = unifint(diff_lb, diff_ub, (1, h//3))
    objw = unifint(diff_lb, diff_ub, (1, w//3))
    bounds = asindices(canvas(-1, (objh, objw)))
    sp = choice(totuple(bounds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    gi = canvas(bgc, (h, w))
    objc = choice(ccols)
    gi = fill(gi, objc, obj)
    sep = connect((objh+1, 0), (objh+1, objw+1)) | connect((0, objw+1), (objh+1, objw+1))
    gi = fill(gi, sepc, sep)
    inds = asindices(gi)
    inds -= backdrop(sep)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    baseobj = normalize(obj)
    obj = normalize(obj)
    go = tuple(e for e in gi)
    while (succ < nobjs and tr < maxtr) or succ == 0:
        tr += 1
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            col = choice(ccols)
            gi = fill(gi, col, plcd)
            go = fill(go, sepc if succ == 0 else col, plcd)
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
        objh = randint(1, h // 3)
        objw = randint(2 if objh == 1 else 1, w // 3)
        if choice((True, False)):
            objh, objw = objw, objh
        bounds = asindices(canvas(-1, (objh, objw)))
        sp = choice(totuple(bounds))
        obj = {sp}
        ncells = unifint(diff_lb, diff_ub, (1, (objh * objw)))
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        obj = set(obj)
        if obj == baseobj:
            if len(obj) < objh * objw:
                obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
            else:
                obj = remove(choice(totuple(corners(obj))), obj)
        obj = normalize(obj)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_e5062a87(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    eligcol, objc = sample(cols, 2)
    gi = canvas(eligcol, (h, w))
    inds = asindices(gi)
    sp = choice(totuple(inds))
    obj = {sp}
    ncells = unifint(diff_lb, diff_ub, (3, 9))
    for k in range(ncells - 1):
        obj.add(choice(totuple((inds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    nnoise = unifint(diff_lb, diff_ub, (int(0.2*h*w), int(0.5*h*w)))
    locs = sample(totuple(inds), nnoise)
    gi = fill(gi, 0, locs)
    noccs = unifint(diff_lb, diff_ub, (2, max(2, (h * w) // (len(obj) * 3))))
    oh, ow = shape(obj)
    for k in range(noccs):
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        loc = (loci, locj)
        gi = fill(gi, objc if k == noccs - 1 else 0, shift(obj, loc))
    occs = occurrences(gi, recolor(0, obj))
    res = mapply(lbind(shift, obj), occs)
    go = fill(gi, objc, res)
    return {'input': gi, 'output': go}


def generate_bc1d5164(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (2, 14))
    fullh = 2 * h - 1
    fullw = 2 * w + 1
    bgc, objc = sample(cols, 2)
    inds = asindices(canvas(-1, (h, w)))
    nA = randint(1, (h - 1) * (w - 1) - 1)
    nB = randint(1, (h - 1) * (w - 1) - 1)
    nC = randint(1, (h - 1) * (w - 1) - 1)
    nD = randint(1, (h - 1) * (w - 1) - 1)
    A = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] < w - 1)), nA)
    B = sample(totuple(sfilter(inds, lambda ij: ij[0] < h - 1 and ij[1] > 0)), nB)
    C = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] < w - 1)), nC)
    D = sample(totuple(sfilter(inds, lambda ij: ij[0] > 0 and ij[1] > 0)), nD)
    gi = canvas(bgc, (fullh, fullw))
    gi = fill(gi, objc, A)
    gi = fill(gi, objc, shift(B, (0, fullw - w)))
    gi = fill(gi, objc, shift(C, (fullh - h, 0)))
    gi = fill(gi, objc, shift(D, (fullh - h, fullw - w)))
    go = canvas(bgc, (h, w))
    go = fill(go, objc, set(A) | set(B) | set(C) | set(D))
    return {'input': gi, 'output': go}


def generate_11852cab(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    r1 = ((0, 0), (0, 4), (4, 0), (4, 4))
    r2 = ((2, 0), (0, 2), (4, 2), (2, 4))
    r3 = ((1, 1), (3, 1), (1, 3), (3, 3))
    r4 = ((2, 2),)
    rings = [r4, r3, r2, r1]
    bx = backdrop(frozenset(r1))
    h = unifint(diff_lb, diff_ub, (7, 30))
    w = unifint(diff_lb, diff_ub, (7, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = shift(asindices(trim(gi)), UNITY)
    nobjs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // 36)))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    while succ < nobjs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 5 and ij[0] <= w - 5)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(bx, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - outbox(plcd)
            ringcols = [choice(ccols) for k in range(4)]
            plcdrings = [shift(r, loc) for r in rings]
            gi = fill(gi, ringcols[0], plcdrings[0])
            go = fill(go, ringcols[0], plcdrings[0])
            idx = randint(1, 3)
            gi = fill(gi, ringcols[idx], plcdrings[idx])
            go = fill(go, ringcols[idx], plcdrings[idx])
            remrings = plcdrings[1:idx] + plcdrings[idx+1:]
            remringcols = ringcols[1:idx] + ringcols[idx+1:]
            numrs = unifint(diff_lb, diff_ub, (1, 2))
            locs = sample((0, 1), numrs)
            remrings = [rr for j, rr in enumerate(remrings) if j in locs]
            remringcols = [rr for j, rr in enumerate(remringcols) if j in locs]
            tofillgi = merge(frozenset(
                recolor(col, frozenset(sample(totuple(remring), 4 - unifint(diff_lb, diff_ub, (0, 3))))) for remring, col in zip(remrings, remringcols)
            ))
            tofillgo = merge(frozenset(
                recolor(col, remring) for remring, col in zip(remrings, remringcols)
            ))
            if min(shape(tofillgi)) == 5:
                succ += 1
                gi = paint(gi, tofillgi)
                go = paint(go, tofillgo)
    return {'input': gi, 'output': go}


def generate_025d127b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(3, 6)
        ow = randint(3, 6)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        topl = connect((0, 0), (0, ow - 1))
        leftl = connect((1, 0), (oh - 2, oh - 3))
        rightl = connect((1, ow), (oh - 2, ow + oh - 3))
        botl = connect((oh - 1, oh - 2), (oh - 1, oh - 3 + ow))
        inobj = topl | leftl | rightl | botl
        outobj = shift(topl, (0, 1)) | botl | shift(leftl, (0, 1)) | connect((1, ow+1), (oh - 3, ow + oh - 3)) | {(oh - 2, ow + oh - 3)}
        outobj = sfilter(outobj, lambda ij: ij[1] <= rightmost(inobj))
        fullobj = inobj | outobj
        inobj = shift(inobj, loc)
        outobj = shift(outobj, loc)
        fullobj = shift(fullobj, loc)
        if fullobj.issubset(inds):
            inds = (inds - fullobj) - mapply(neighbors, fullobj)
            succ += 1
            col = choice(ccols)
            gi = fill(gi, col, inobj)
            go = fill(go, col, outobj)
    return {'input': gi, 'output': go}


def generate_045e512c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (11, 30))
    w = unifint(diff_lb, diff_ub, (11, 30))
    while True:
        oh = unifint(diff_lb, diff_ub, (2, min(4, (h - 2) // 3)))
        ow = unifint(diff_lb, diff_ub, (2, min(4, (w - 2) // 3)))
        bounds = asindices(canvas(-1, (oh, ow)))
        c1 = choice(totuple(connect((0, 0), (oh - 1, 0))))
        c2 = choice(totuple(connect((0, 0), (0, ow - 1))))
        c3 = choice(totuple(connect((oh - 1, ow - 1), (oh - 1, 0))))
        c4 = choice(totuple(connect((oh - 1, ow - 1), (0, ow - 1))))
        obj = {c1, c2, c3, c4}
        remcands = totuple(bounds - obj)
        ncells = unifint(diff_lb, diff_ub, (0, len(remcands)))
        for k in range(ncells):
            loc = choice(remcands)
            obj.add(loc)
            remcands = remove(loc, remcands)
        objt = normalize(obj)
        cc = canvas(0, shape(obj))
        cc = fill(cc, 1, objt)
        if len(colorfilter(objects(cc, T, T, F), 1)) == 1:
            break
    loci = randint(oh + 1, h - 2 * oh - 1)
    locj = randint(ow + 1, w - 2 * ow - 1)
    loc = (loci, locj)
    bgc, objc = sample(cols, 2)
    remcols = remove(bgc, remove(objc, cols))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    obj = shift(recolor(objc, obj), loc)
    gi = paint(gi, obj)
    go = paint(go, obj)
    options = totuple(neighbors((0, 0)))
    ndirs = unifint(diff_lb, diff_ub, (1, 8))
    dirs = sample(options, ndirs)
    dcols = [choice(ccols) for k in range(ndirs)]
    hbars = hfrontier((loci - 2, 0)) | hfrontier((loci+oh+1, 0))
    vbars = vfrontier((0, locj - 2)) | vfrontier((0, locj+ow+1))
    bars = hbars | vbars
    ofs = increment((oh, ow))
    for direc, col in zip(dirs, dcols):
        indicatorobj = shift(obj, multiply(direc, increment((oh, ow))))
        indicatorobj = sfilter(indicatorobj, lambda cij: cij[1] in bars)
        nindsd = unifint(diff_lb, diff_ub, (0, len(indicatorobj) - 1))
        ninds = len(indicatorobj) - nindsd
        indicatorobj = set(sample(totuple(indicatorobj), ninds))
        if len(indicatorobj) > 0 and len(indicatorobj) < len(obj):
            gi = fill(gi, col, indicatorobj)
            for k in range(1, 10):
                go = fill(go, col, shift(obj, multiply(multiply(k, direc), ofs)))
    return {'input': gi, 'output': go}


def generate_1b60fb0c(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    odh = unifint(diff_lb, diff_ub, (2, min(h, w)//2))
    loci = randint(0, h - 2 * odh)
    locj = randint(0, w - 2 * odh)
    loc = (loci, locj)
    bgc, objc = sample(cols, 2)
    quad = canvas(bgc, (odh, odh))
    ncellsd = unifint(diff_lb, diff_ub, (0, odh ** 2 // 2))
    ncells = choice((ncellsd, odh ** 2 - ncellsd))
    ncells = min(max(1, ncells), odh ** 2 - 1)
    cells = sample(totuple(asindices(canvas(-1, (odh, odh)))), ncells)
    g1 = fill(quad, objc, cells)
    g2 = rot90(g1)
    g3 = rot90(g2)
    g4 = rot90(g3)
    c1 = shift(ofcolor(g1, objc), (0, 0))
    c2 = shift(ofcolor(g2, objc), (0, odh))
    c3 = shift(ofcolor(g3, objc), (odh, odh))
    c4 = shift(ofcolor(g4, objc), (odh, 0))
    shftamt = randint(0, odh)
    c1 = shift(c1, (0, shftamt))
    c2 = shift(c2, (shftamt, 0))
    c3 = shift(c3, (0, -shftamt))
    c4 = shift(c4, (-shftamt, 0))
    cs = (c1, c2, c3, c4)
    rempart = choice(cs)
    inobjparts = remove(rempart, cs)
    inobj = merge(set(inobjparts))
    rempart = rempart - inobj
    inobj = shift(inobj, loc)
    rempart = shift(rempart, loc)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, inobj)
    go = fill(gi, 2, rempart)
    return {'input': gi, 'output': go}


def generate_1f0c79e5(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, objc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 24))
    inds = asindices(gi)
    obj = ((0, 0), (0, 1), (1, 0), (1, 1))
    for k in range(nobjs):
        cands = sfilter(inds, lambda ij: shift(set(obj), ij).issubset(inds))
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        nred = unifint(diff_lb, diff_ub, (1, 3))
        reds = sample(totuple(plcd), nred)
        gi = fill(gi, objc, plcd)
        gi = fill(gi, 2, reds)
        for idx in reds:
            direc = decrement(multiply(2, add(idx, invert(loc))))
            go = fill(go, objc, mapply(rbind(shoot, direc), frozenset(plcd)))
        inds = (inds - plcd) - mapply(dneighbors, set(plcd))
    return {'input': gi, 'output': go}


def generate_1f876c06(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nlns = unifint(diff_lb, diff_ub, (1, min(min(h, w), 9)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nlns)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    direcs = ineighbors((0, 0))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        lns = []
        for direc in direcs:
            ln = [loc]
            ofs = 1
            while True:
                nextpix = add(loc, multiply(ofs, direc))
                ofs += 1
                if nextpix not in inds:
                    break
                ln.append(nextpix)
            if len(ln) > 2:
                lns.append(ln)
        if len(lns) > 0:
            succ += 1
            lns = sorted(lns, key=len)
            idx = unifint(diff_lb, diff_ub, (0, len(lns) - 1))
            ln = lns[idx]
            col = ccols[0]
            ccols = ccols[1:]
            gi = fill(gi, col, {ln[0], ln[-1]})
            go = fill(go, col, set(ln))
            inds = inds - set(ln)
    return {'input': gi, 'output': go}


def generate_22233c11(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    succ = 0
    tr = 0
    maxtr = 10 * nobjs
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    while succ < nobjs and tr < maxtr:
        if len(inds) == 0:
            break
        tr += 1
        od = randint(1, 3)
        fulld = 4 * od
        g = canvas(bgc, (4, 4))
        g = fill(g, 8, {(0, 3), (3, 0)})
        col = choice(ccols)
        g = fill(g, col, {(1, 1), (2, 2)})
        if choice((True, False)):
            g = hmirror(g)
        g = upscale(g, od)
        inobj = recolor(col, ofcolor(g, col))
        outobj = inobj | recolor(8, ofcolor(g, 8))
        loc = choice(totuple(inds))
        outobj = shift(outobj, loc)
        inobj = shift(inobj, loc)
        outobji = toindices(outobj)
        if toindices(inobj).issubset(inds) and (outobji & fullinds).issubset(inds):
            succ += 1
            inds = (inds - outobji) - mapply(neighbors, outobji)
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {'input': gi, 'output': go}


def generate_264363fd(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    cp = (2, 2)
    neighs = neighbors(cp)
    o1 = shift(frozenset({(0, 1), (-1, 1)}), (1, 1))
    o2 = shift(frozenset({(1, 0), (1, -1)}), (1, 1))
    o3 = shift(frozenset({(2, 1), (3, 1)}), (1, 1))
    o4 = shift(frozenset({(1, 2), (1, 3)}), (1, 1))
    mpr = {o1: (-1, 0), o2: (0, -1), o3: (1, 0), o4: (0, 1)}
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    bgc, sqc, linc = sample(cols, 3)
    remcols = difference(cols, (bgc, sqc, linc))
    cpcol = choice(remcols)
    nbhcol = choice(remcols)
    nspikes = randint(1, 4)
    spikes = sample((o1, o2, o3, o4), nspikes)
    lns = merge(set(spikes))
    obj = {(cpcol, cp)} | recolor(linc, lns) | recolor(nbhcol, neighs - lns)
    loci = randint(0, h - 5)
    locj = randint(0, w - 5)
    loc = (loci, locj)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = paint(gi, shift(obj, loc))
    numsq = unifint(diff_lb, diff_ub, (1, (h * w) // 100))
    succ = 0
    tr = 0
    maxtr = 10 * numsq
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(shift(obj, loc)))
    while succ < numsq and tr < maxtr:
        tr += 1
        gh = randint(5, h//2+1)
        gw = randint(5, w//2+1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - gh and ij[1] <= w - gw)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        g1 = canvas(sqc, (gh, gw))
        g2 = canvas(sqc, (gh, gw))
        ginds = asindices(g1)
        gindsfull = asindices(g1)
        bck = shift(ginds, loc)
        if bck.issubset(inds):
            noccs = unifint(diff_lb, diff_ub, (1, (gh * gw) // 25))
            succ2 = 0
            tr2 = 0
            maxtr2 = 5 * noccs
            while succ2 < noccs and tr2 < maxtr2:
                tr2 += 1
                cands2 = sfilter(ginds, lambda ij: ij[0] <= gh - 5 and ij[1] <= gw - 5)
                if len(cands2) == 0:
                    break
                loc2 = choice(totuple(cands2))
                lns2 = merge(frozenset({shoot(add(cp, add(loc2, mpr[spike])), mpr[spike]) for spike in spikes}))
                lns2 = lns2 & gindsfull
                plcd2 = shift(obj, loc2)
                plcd2i = toindices(plcd2)
                if plcd2i.issubset(ginds) and lns2.issubset(ginds | ofcolor(g2, linc)) and len(lns2 - plcd2i) > 0:
                    succ2 += 1
                    ginds = ((ginds - plcd2i) - mapply(neighbors, plcd2i)) - lns2
                    g1 = fill(g1, cpcol, {add(cp, loc2)})
                    g2 = paint(g2, plcd2)
                    g2 = fill(g2, linc, lns2)
            if succ2 > 0:
                succ += 1
                inds = (inds - bck) - outbox(bck)
                objfull1 = shift(asobject(g1), loc)
                objfull2 = shift(asobject(g2), loc)
                gi = paint(gi, objfull1)
                go = paint(go, objfull2)
    return {'input': gi, 'output': go}


def generate_29ec7d0e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h//hp+1):
        for b in range(w//wp+1):
            loci = (a+1) + hp * a
            locj = (b+1) + wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_3bd67248(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 4))
    h = unifint(diff_lb, diff_ub, (3, 15))
    w = unifint(diff_lb, diff_ub, (3, 15))
    bgc, linc = sample(cols, 2)
    fac = unifint(diff_lb, diff_ub, (1, 30 // max(h, w)))
    gi = canvas(bgc, (h, w))
    gi = fill(gi, linc, connect((0, 0), (h - 1, 0)))
    go = fill(gi, 4, connect((h - 1, 1), (h - 1, w - 1)))
    go = fill(go, 2, shoot((h - 2, 1), (-1, 1)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    gi = upscale(gi, fac)
    go = upscale(go, fac)
    return {'input': gi, 'output': go}


def generate_484b58aa(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    noisec = choice(cols)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(-1, (h, w))
    locs = set()
    ofs = randint(1, hp - 1)
    for a in range(2*(h//hp+1)):
        for b in range(w//wp+1):
            loci = hp * a - ofs * b
            locj = wp * b
            locs.add((loci, locj))
            go = paint(go, shift(pobj, (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        if pobj in apply(normalize, apply(rbind(toobject, gi2), places)):
            if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2:
                succ += 1
                gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_6aa20dc0(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    od = unifint(diff_lb, diff_ub, (2, 4))
    ncellsextra = randint(1, max(1, (od ** 2 - 2) // 2))
    sinds = asindices(canvas(-1, (od, od)))
    extracells = set(sample(totuple(sinds - {(0, 0), (od - 1, od - 1)}), ncellsextra))
    extracells.add(choice(totuple(dneighbors((0, 0)) & sinds)))
    extracells.add(choice(totuple(dneighbors((od - 1, od - 1)) & sinds)))
    extracells = frozenset(extracells)
    bgc, fgc, c1, c2 = sample(cols, 4)
    obj = frozenset({(c1, (0, 0)), (c2, (od - 1, od - 1))}) | recolor(fgc, extracells)
    obj = obj | dmirror(obj)
    if choice((True, False)):
        obj = hmirror(obj)
    gi = canvas(bgc, (h, w))
    loci = randint(0, h - od)
    locj = randint(0, w - od)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = asindices(gi)
    inds = inds - backdrop(outbox(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (od ** 2 * 2))))
    succ = 0
    tr = 0
    maxtr = 4 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        mf1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        mf = compose(mf2, mf1)
        cobj = normalize(upscale(mf(obj), fac))
        ohx, owx = shape(cobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - ohx and ij[1] <= w - owx)
        if len(cands) == 0:
            continue
        locc = choice(totuple(cands))
        cobjo = shift(cobj, locc)
        cobji = sfilter(cobjo, lambda cij: cij[0] != fgc)
        cobjoi = toindices(cobjo)
        if cobjoi.issubset(inds):
            succ += 1
            inds = inds - backdrop(outbox(cobjoi))
            gi = paint(gi, cobji)
            go = paint(go, cobjo)
    return {'input': gi, 'output': go}


def generate_6855a6e4(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(sample(totuple(cands), ncells))
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    return {'input': gi, 'output': go}


def generate_39a8645d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    oh = randint(2, 4)
    ow = randint(2, 4)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, oh + ow))
    ccols = sample(remcols, nobjs+1)
    mxcol = ccols[0]
    rcols = ccols[1:]
    maxnocc = unifint(diff_lb, diff_ub, (nobjs + 2, max(nobjs + 2, (h * w) // 16)))
    tr = 0
    maxtr = 10 * maxnocc
    succ = 0
    allobjs = []
    bounds = asindices(canvas(-1, (oh, ow)))
    for k in range(nobjs + 1):
        while True:
            ncells = randint(oh + ow - 1, oh * ow)
            cobj = {choice(totuple(bounds))}
            while shape(cobj) != (oh, ow) and len(cobj) < ncells:
                cobj.add(choice(totuple((bounds - cobj) & mapply(neighbors, cobj))))
            if cobj not in allobjs:
                break
        allobjs.append(frozenset(cobj))
    mcobj = normalize(allobjs[0])
    remobjs = apply(normalize, allobjs[1:])
    mxobjcounter = 0
    remobjcounter = {robj: 0 for robj in remobjs}
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    while tr < maxtr and succ < maxnocc:
        tr += 1
        candobjs = [robj for robj, cnt in remobjcounter.items() if cnt + 1 < mxobjcounter]
        if len(candobjs) == 0 or randint(0, 100) / 100 > diff_lb:
            obj = mcobj
            col = mxcol
        else:
            obj = choice(candobjs)
            col = rcols[remobjs.index(obj)]
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds - mapply(neighbors, ofcolor(gi, col))):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, col, plcd)
            if obj in remobjcounter:
                remobjcounter[obj] += 1
            else:
                mxobjcounter += 1
    go = fill(canvas(bgc, shape(mcobj)), mxcol, mcobj)
    return {'input': gi, 'output': go}


def generate_150deff5(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 8))
    bo = {(0, 0), (0, 1), (1, 0), (1, 1)}
    ro1 = {(0, 0), (0, 1), (0, 2)}
    ro2 = {(0, 0), (1, 0), (2, 0)}
    boforb = set()
    reforb = set()
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // 10))
    inds = asindices(gi)
    needsbgc = []
    for k in range(noccs):
        obj, col = choice(((bo, 8), (choice((ro1, ro2)), 2)))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and shift(obj, ij).issubset(inds))
        if col == 8:
            cands = sfilter(cands, lambda ij: ij not in boforb)
        else:
            cands = sfilter(cands, lambda ij: ij not in reforb)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        if col == 8:
            boforb.add(add(loc, (-2, 0)))
            boforb.add(add(loc, (2, 0)))
            boforb.add(add(loc, (0, 2)))
            boforb.add(add(loc, (0, -2)))
        if col == 2:
            if obj == ro1:
                reforb.add(add(loc, (0, 3)))
                reforb.add(add(loc, (0, -3)))
            else:
                reforb.add(add(loc, (1, 0)))
                reforb.add(add(loc, (-1, 0)))
        plcd = shift(obj, loc)
        gi = fill(gi, fgc, plcd)
        go = fill(go, col, plcd)
        inds = inds - plcd
    return {'input': gi, 'output': go}


def generate_239be575(diff_lb: float, diff_ub: float) -> dict:
    sq = {(0, 0), (1, 1), (0, 1), (1, 0)}
    cols = interval(1, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (6, 30))
        w = unifint(diff_lb, diff_ub, (6, 30))
        c = canvas(0, (h, w))
        fullcands = totuple(asindices(canvas(0, (h - 1, w - 1))))
        a = choice(fullcands)
        b = choice(remove(a, fullcands))
        mindist = unifint(diff_lb, diff_ub, (3, min(h, w) - 3))
        while not manhattan({a}, {b}) > mindist:
            a = choice(fullcands)
            b = choice(remove(a, fullcands))
        markcol, sqcol = sample(cols, 2)
        aset = shift(sq, a)
        bset = shift(sq, b)
        gi = fill(c, sqcol, aset | bset)
        cands = totuple(ofcolor(gi, 0))
        num = unifint(diff_lb, diff_ub, (int(0.25 * len(cands)), int(0.75 * len(cands))))
        mc = sample(cands, num)
        gi = fill(gi, markcol, mc)
        bobjs = colorfilter(objects(gi, T, F, F), markcol)
        ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        shoudlhaveconn = choice((True, False))
        if shoudlhaveconn and len(ss) == 0:
            while len(ss) == 0:
                opts2 = totuple(ofcolor(gi, 0))
                if len(opts2) == 0:
                    break
                gi = fill(gi, markcol, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        elif not shoudlhaveconn and len(ss) > 0:
            while len(ss) > 0:
                opts2 = totuple(ofcolor(gi, markcol))
                if len(opts2) == 0:
                    break
                gi = fill(gi, 0, {choice(opts2)})
                bobjs = colorfilter(objects(gi, T, F, F), markcol)
                ss = sfilter(bobjs, fork(both, rbind(adjacent, aset), rbind(adjacent, bset)))
        if len(palette(gi)) == 3:
            break
    oc = markcol if shoudlhaveconn else 0
    go = canvas(oc, (1, 1))
    return {'input': gi, 'output': go}


def generate_0dfd9992(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    hp = unifint(diff_lb, diff_ub, (2, h//2-1))
    wp = unifint(diff_lb, diff_ub, (2, w//2-1))
    pinds = asindices(canvas(-1, (hp, wp)))
    bgc, noisec = sample(cols, 2)
    remcols = remove(noisec, cols)
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, numc)
    pobj = frozenset({(choice(ccols), ij) for ij in pinds})
    go = canvas(bgc, (h, w))
    locs = set()
    for a in range(h//hp+1):
        for b in range(w//wp+1):
            loci = hp * a
            locj = wp * b
            locs.add((loci, locj))
            mf1 = identity if a % 2 == 0 else hmirror
            mf2 = identity if b % 2 == 0 else vmirror
            mf = compose(mf1, mf2)
            go = paint(go, shift(mf(pobj), (loci, locj)))
    numpatches = unifint(diff_lb, diff_ub, (1, int((h * w) ** 0.5 // 2)))
    gi = tuple(e for e in go)
    places = apply(lbind(shift, pinds), locs)
    succ = 0
    tr = 0
    maxtr = 5 * numpatches
    while succ < numpatches and tr < maxtr:
        tr += 1
        ph = randint(2, 6)
        pw = randint(2, 6)
        loci = randint(0, h - ph)
        locj = randint(0, w - pw)
        ptch = backdrop(frozenset({(loci, locj), (loci + ph - 1, locj + pw - 1)}))
        gi2 = fill(gi, noisec, ptch)
        candset = apply(normalize, apply(rbind(toobject, gi2), places))
        if len(sfilter(gi2, lambda r: noisec not in r)) >= 2 and len(sfilter(dmirror(gi2), lambda r: noisec not in r)) >= 2 and (pobj in candset or hmirror(pobj) in candset or vmirror(pobj) in candset or hmirror(vmirror(pobj)) in candset):
            succ += 1
            gi = gi2
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_d06dbe63(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    obj1 = mapply(lbind(shift, frozenset({(-1, 0), (-2, 0), (-2, 1), (-2, 2)})), {(-k * 2, 2 * k) for k in range(15)})
    obj2 = mapply(lbind(shift, frozenset({(1, 0), (2, 0), (2, -1), (2, -2)})), {(2 * k, -k * 2) for k in range(15)})
    obj = obj1 | obj2
    objf = lambda ij: shift(obj, ij)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    ndots = unifint(diff_lb, diff_ub, (1, min(h, w)))
    succ = 0
    tr = 0
    maxtr = 4 * ndots
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < ndots:
        tr += 1
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        objx = objf(loc)
        if (objx & fullinds).issubset(inds):
            succ += 1
            inds = (inds - objx) - {loc}
            gi = fill(gi, dotc, {loc})
            go = fill(go, dotc, {loc})
            go = fill(go, 5, objx)
    return {'input': gi, 'output': go}


def generate_a3325580(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, 9))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ccols = sample(remcols, nobjs)
    gi = canvas(bgc, (h, w))
    lmocc = set()
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 4 * nobjs
    seenobjs = set()
    mxncells = randint(nobjs+1, 30)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(1, 6)
        ow = randint(1, 6)
        while oh * ow < mxncells:
            oh = randint(1, 6)
            ow = randint(1, 6)
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = randint(1, oh * ow)
        ncells = unifint(diff_lb, diff_ub, (1, min(oh * ow, mxncells)))
        ncells = unifint(diff_lb, diff_ub, (ncells, min(oh * ow, mxncells)))
        sp = choice(totuple(bounds))
        obj = {sp}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        if obj in seenobjs:
            continue
        obj = normalize(obj)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow and ij[1] not in lmocc)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            gi = fill(gi, ccols[succ], plcd)
            succ += 1
            lmocc.add(loc[1])
    objs = objects(gi, T, F, T)
    mxncells = valmax(objs, size)
    objs = sfilter(objs, matcher(size, mxncells))
    objs = order(objs, leftmost)
    go = canvas(-1, (mxncells, len(objs)))
    for idx, o in enumerate(objs):
        go = fill(go, color(o), connect((0, idx), (mxncells - 1, idx)))
    return {'input': gi, 'output': go}


def generate_1fad071e(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nbl = randint(0, 5)
    nobjs = unifint(diff_lb, diff_ub, (nbl, max(nbl, (h * w) // 10)))
    bgc, otherc = sample(cols, 2)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    bcount = 0
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    ofcfrbinds = {1: set(), otherc: set()}
    while succ < nobjs and tr < maxtr:
        tr += 1
        col = choice((1, otherc))
        oh = randint(1, 3)
        ow = randint(1, 3)
        if bcount < nbl:
            col = 1
            oh, ow = 2, 2
        else:
            while col == 1 and oh == ow == 2:
                col = choice((1, otherc))
                oh = randint(1, 3)
                ow = randint(1, 3)
        bd = backdrop(frozenset({(0, 0), (oh - 1, ow - 1)}))
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = shift(bd, loc)
        if bd.issubset(inds) and len(mapply(dneighbors, bd) & ofcfrbinds[col]) == 0:
            succ += 1
            inds = inds - bd
            ofcfrbinds[col] = ofcfrbinds[col] | mapply(dneighbors, bd) | bd
            gi = fill(gi, col, bd)
            if col == 1 and oh == ow == 2:
                bcount += 1
    go = (repeat(1, bcount) + repeat(bgc, 5 - bcount),)
    return {'input': gi, 'output': go}


def generate_27a28665(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    mapping = [
    (1, {(0, 0), (0, 1), (1, 0), (1, 2), (2, 1)}),
    (2, {(0, 0), (1, 1), (2, 0), (0, 2), (2, 2)}),
    (3, {(2, 0), (0, 1), (0, 2), (1, 1), (1, 2)}),
    (6, {(1, 1), (0, 1), (1, 0), (1, 2), (2, 1)})
    ]
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    col, obj = choice(mapping)
    bgc, objc = sample(cols, 2)
    fac = unifint(diff_lb, diff_ub, (1, min(h, w) // 3))
    go = canvas(col, (1, 1))
    gi = canvas(bgc, (h, w))
    canv = canvas(bgc, (3, 3))
    canv = fill(canv, objc, obj)
    canv = upscale(canv, fac)
    obj = asobject(canv)
    loci = randint(0, h - 3 * fac)
    locj = randint(0, w - 3 * fac)
    loc = (loci, locj)
    gi = paint(gi, shift(obj, loc))
    return {'input': gi, 'output': go}


def generate_b775ac94(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    gi = canvas(0, (1, 1))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 25))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 5)
        ow = randint(2, 5)
        canv = canvas(bgc, (oh, ow))
        c1, c2, c3, c4 = sample(remcols, 4)
        obj = {(0, 0)}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow - 1)
        bounds = asindices(canv)
        for k in range(ncells):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        gLR = fill(canv, c1, obj)
        gLL = replace(vmirror(gLR), c1, c2)
        gUR = replace(hmirror(gLR), c1, c3)
        gUL = replace(vmirror(hmirror(gLR)), c1, c4)
        gU = hconcat(gUL, gUR)
        gL = hconcat(gLL, gLR)
        g = vconcat(gU, gL)
        g2 = canvas(bgc, (oh * 2, ow * 2))
        g2 = fill(g2, c1, shift(obj, (oh, ow)))
        nkeepcols = unifint(diff_lb, diff_ub, (1, 3))
        keepcols = sample((c2, c3, c4), nkeepcols)
        for cc in (c2, c3, c4):
            if cc not in keepcols:
                g = replace(g, cc, bgc)
            else:
                ofsi = -1 if cc in (c3, c4) else 0
                ofsj = -1 if cc in (c2, c4) else 0
                g2 = fill(g2, cc, {(oh + ofsi, ow + ofsj)})
        rotf = choice((identity, rot90, rot180, rot270))
        g = rotf(g)
        g2 = rotf(g2)
        obji = asobject(g2)
        objo = asobject(g)
        objo = sfilter(objo, lambda cij: cij[0] != bgc)
        obji = sfilter(obji, lambda cij: cij[0] != bgc)
        tonorm = invert(ulcorner(objo))
        obji = shift(obji, tonorm)
        objo = shift(objo, tonorm)
        oh, ow = shape(objo)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcdi = shift(obji, loc)
        plcdo = shift(objo, loc)
        plcdoi = toindices(plcdo)
        if plcdoi.issubset(inds):
            succ += 1
            inds = (inds - plcdoi) - mapply(neighbors, plcdoi)
            gi = paint(gi, plcdi)
            go = paint(go, plcdo)
    return {'input': gi, 'output': go}


def generate_6f8cd79b(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (0, h * w))
    inds = asindices(gi)
    cells = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in cells}
    gi = paint(gi, obj)
    brd = box(inds)
    go = fill(gi, 8, brd)
    return {'input': gi, 'output': go}


def generate_de1cd16c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    noisec = choice(cols)
    remcols = remove(noisec, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    starterc = ccols[0]
    ccols = ccols[1:]
    gi = canvas(starterc, (h, w))
    for k in range(ncols - 1):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: height(o) > 5 or width(o) > 5)
        if len(objs) == 0:
            break
        objs = totuple(objs)
        obj = choice(objs)
        if height(obj) > 5 and width(obj) > 5:
            ax = choice((0, 1))
        elif height(obj) > 5:
            ax = 0
        elif width(obj) > 5:
            ax = 1
        if ax == 0:
            loci = randint(uppermost(obj)+3, lowermost(obj)-2)
            newobj = sfilter(toindices(obj), lambda ij: ij[0] >= loci)
        elif ax == 1:
            locj = randint(leftmost(obj)+3, rightmost(obj)-2)
            newobj = sfilter(toindices(obj), lambda ij: ij[1] >= locj)
        gi = fill(gi, ccols[k], newobj)
    objs = order(objects(gi, T, F, F), size)
    allowances = [max(1, ((height(o) - 2) * (width(o) - 2)) // 2) for o in objs]
    meann = max(1, int(sum(allowances) / len(allowances)))
    chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    while max(chosens) == 0:
        chosens = [randint(0, min(meann, allowed)) for allowed in allowances]
    mx = max(chosens)
    fixinds = [idx for idx, cnt in enumerate(chosens) if cnt == mx]
    gogoind = fixinds[0]
    gogocol = color(objs[gogoind])
    fixinds = fixinds[1:]
    for idx in fixinds:
        chosens[idx] -= 1
    for obj, cnt in zip(objs, chosens):
        locs = sample(totuple(backdrop(inbox(toindices(obj)))), cnt)
        gi = fill(gi, noisec, locs)
    go = canvas(gogocol, (1, 1))
    return {'input': gi, 'output': go}


def generate_6cf79266(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nfgcs = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(cols, nfgcs)
    gi = canvas(-1, (h, w))
    fgcobj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, fgcobj)
    num = unifint(diff_lb, diff_ub, (int(0.25 * h * w), int(0.6 * h * w)))
    inds = asindices(gi)
    locs = sample(totuple(inds), num)
    gi = fill(gi, 0, locs)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    cands = asindices(canvas(-1, (h - 2, w - 2)))
    locs = sample(totuple(cands), noccs)
    mini = asindices(canvas(-1, (3, 3)))
    for ij in locs:
        gi = fill(gi, 0, shift(mini, ij))
    trg = recolor(0, mini)
    occs = occurrences(gi, trg)
    go = tuple(e for e in gi)
    for occ in occs:
        go = fill(go, 1, shift(mini, occ))
    return {'input': gi, 'output': go}


def generate_a85d4709(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3, 4))
    h = unifint(diff_lb, diff_ub, (2, 30))
    w3 = unifint(diff_lb, diff_ub, (1, 10))
    w = w3 * 3
    bgc, dotc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    for ii in range(h):
        loc = randint(0, w3 - 1)
        dev = unifint(diff_lb, diff_ub, (0, w3 // 2 + 1))
        loc = w3 // 3 + choice((+dev, -dev))
        loc = min(max(0, loc), w3 - 1)
        ofs, col = choice(((0, 2), (1, 4), (2, 3)))
        loc += ofs * w3
        gi = fill(gi, dotc, {(ii, loc)})
        ln = connect((ii, 0), (ii, w - 1))
        go = fill(go, col, ln)
    return {'input': gi, 'output': go}


def generate_f8a8fe49(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    fullh = unifint(diff_lb, diff_ub, (10, h))
    fullw = unifint(diff_lb, diff_ub, (3, w))
    bgc, objc, boxc = sample(cols, 3)
    bcanv = canvas(bgc, (h, w))
    loci = randint(0, h - fullh)
    locj = randint(0, w - fullw)
    loc = (loci, locj)
    canvi = canvas(bgc, (fullh, fullw))
    canvo = canvas(bgc, (fullh, fullw))
    objh = (fullh // 2 - 3) // 2
    br = connect((objh + 1, 0), (objh + 1, fullw - 1))
    br = br | {(objh + 2, 0), (objh + 2, fullw - 1)}
    cands = backdrop(frozenset({(0, 1), (objh - 1, fullw - 2)}))
    for k in range(2):
        canvi = fill(canvi, boxc, br)
        canvo = fill(canvo, boxc, br)
        ncellsd = unifint(diff_lb, diff_ub, (0, (objh * (fullw - 2)) // 2))
        ncells = choice((ncellsd, objh * (fullw - 2) - ncellsd))
        ncells = min(max(1, ncells), objh * (fullw - 2))
        cells = frozenset(sample(totuple(cands), ncells))
        cells = insert(choice(totuple(sfilter(cands, lambda ij: ij[0] == lowermost(cands)))), cells)
        canvi = fill(canvi, objc, cells)
        canvo = fill(canvo, objc, shift(hmirror(cells), (objh + 3, 0)))
        canvi = hmirror(canvi)
        canvo = hmirror(canvo)
    gi = paint(bcanv, shift(asobject(canvi), loc))
    go = paint(bcanv, shift(asobject(canvo), loc))
    if choice((True, False)):
        gi = dmirror(gi)
        go = dmirror(go)
    go, gi = gi, go
    return {'input': gi, 'output': go}


def generate_f8c80d96(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    ow = randint(1, 3 if h > 10 else 2)
    oh = randint(1, 3 if w > 10 else 2)
    loci = randint(-oh+1, h-1)
    locj = randint(-ow+1, w-1)
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(5, (h, w))
    ln1 = outbox(obj)
    ulci, ulcj = decrement(ulcorner(obj))
    lrci, lrcj = increment(lrcorner(obj))
    hoffs = randint(2, 4 if h > 12 else 3)
    woffs = randint(2, 4 if w > 12 else 3)
    lns = []
    for k in range(max(h, w) // min(hoffs, woffs) + 1):
        lnx = box(frozenset({(ulci - hoffs * k, ulcj - woffs * k), (lrci + hoffs * k, lrcj + woffs * k)}))
        lns.append(lnx)
    inds = asindices(gi)
    lns = sfilter(lns, lambda ln: len(ln & inds) > 0)
    nlns = len(lns)
    nmissing = unifint(diff_lb, diff_ub, (0, nlns - 2))
    npresent = nlns - nmissing
    for k in range(npresent):
        gi = fill(gi, linc, lns[k])
    for ln in lns:
        go = fill(go, linc, ln)
    return {'input': gi, 'output': go}


def generate_f35d900a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(5, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (4, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    bgc, c1, c2 = sample(cols, 3)
    oh = unifint(diff_lb, diff_ub, (4, h))
    ow = unifint(diff_lb, diff_ub, (4, w))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    gi = fill(gi, c1, {ulcorner(bx), lrcorner(bx)})
    gi = fill(gi, c2, {urcorner(bx), llcorner(bx)})
    go = fill(go, c1, {ulcorner(bx), lrcorner(bx)})
    go = fill(go, c2, {urcorner(bx), llcorner(bx)})
    go = fill(go, c1, neighbors(urcorner(bx)) | neighbors(llcorner(bx)))
    go = fill(go, c2, neighbors(ulcorner(bx)) | neighbors(lrcorner(bx)))
    crns = corners(bx)
    for c in crns:
        cobj = {c}
        remcorns = remove(c, crns)
        belongto = sfilter(bx, lambda ij: manhattan(cobj, {ij}) <= valmin(remcorns, lambda cc: manhattan({ij}, {cc})))
        valids = sfilter(belongto, lambda ij: manhattan(cobj, {ij}) > 1 and manhattan(cobj, {ij}) % 2 == 0)
        go = fill(go, 5, valids)
    return {'input': gi, 'output': go}


def generate_ec883f72(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    ohi = unifint(diff_lb, diff_ub, (0, h - 6))
    owi = unifint(diff_lb, diff_ub, (0, w - 6))
    oh = h - 5 - ohi
    ow = w - 5 - owi
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    bgc, sqc, linc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    obj = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    gi = fill(gi, sqc, obj)
    obob = outbox(outbox(obj))
    gi = fill(gi, linc, obob)
    ln1 = shoot(lrcorner(obob), (1, 1))
    ln2 = shoot(ulcorner(obob), (-1, -1))
    ln3 = shoot(llcorner(obob), (1, -1))
    ln4 = shoot(urcorner(obob), (-1, 1))
    lns = (ln1 | ln2 | ln3 | ln4) & ofcolor(gi, bgc)
    go = fill(gi, sqc, lns)
    return {'input': gi, 'output': go}


def generate_ea786f4a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (1, 14))
    w = unifint(diff_lb, diff_ub, (1, 14))
    mp = (h, w)
    h = 2 * h + 1
    w = 2 * w + 1
    linc = choice(cols)
    remcols = remove(linc, cols)
    gi = canvas(linc, (h, w))
    inds = remove(mp, asindices(gi))
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    ln1 = shoot(mp, (-1, -1))
    ln2 = shoot(mp, (1, 1))
    ln3 = shoot(mp, (-1, 1))
    ln4 = shoot(mp, (1, -1))
    go = fill(gi, linc, ln1 | ln2 | ln3 | ln4)
    return {'input': gi, 'output': go}


def generate_ded97339(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, linc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    ndots = unifint(diff_lb, diff_ub, (2, (h * w) // 9))
    inds = asindices(gi)
    dots = set()
    if choice((True, False)):
        idxi = randint(0, h - 1)
        locj1 = randint(0, w - 3)
        locj2 = randint(locj1 + 2, w - 1)
        dots.add((idxi, locj1))
        dots.add((idxi, locj2))
    else:
        idxj = randint(0, w - 1)
        loci1 = randint(0, h - 3)
        loci2 = randint(loci1 + 2, h - 1)
        dots.add((loci1, idxj))
        dots.add((loci2, idxj))
    for k in range(ndots - 2):
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        dots.add(loc)
        inds = (inds - {loc}) - neighbors(loc)
    gi = fill(gi, linc, dots)
    go = tuple(e for e in gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = w - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    go = dmirror(go)
    gi = dmirror(gi)
    for ii, r in enumerate(gi):
        if r.count(linc) > 1:
            a = r.index(linc)
            b = h - r[::-1].index(linc) - 1
            go = fill(go, linc, connect((ii, a), (ii, b)))
    return {'input': gi, 'output': go}


def generate_d687bc17(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    bgc, c1, c2, c3, c4 = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c1, connect((0, 0), (0, w - 1)))
    gi = fill(gi, c2, connect((0, 0), (h - 1, 0)))
    gi = fill(gi, c3, connect((h - 1, w - 1), (0, w - 1)))
    gi = fill(gi, c4, connect((h - 1, w - 1), (h - 1, 0)))
    inds = asindices(gi)
    gi = fill(gi, bgc, corners(inds))
    go = tuple(e for e in gi)
    cands = backdrop(inbox(inbox(inds)))
    ndots = unifint(diff_lb, diff_ub, (1, min(len(cands), h + h + w + w)))
    dots = sample(totuple(cands), ndots)
    dots = {(choice((c1, c2, c3, c4)), ij) for ij in dots}
    n1 = toindices(sfilter(dots, lambda cij: cij[0] == c1))
    n1coverage = apply(last, n1)
    if len(n1coverage) == w - 4 and w > 5:
        n1coverage = remove(choice(totuple(n1coverage)), n1coverage)
    for jj in n1coverage:
        loci = choice([ij[0] for ij in sfilter(n1, lambda ij: ij[1] == jj)])
        gi = fill(gi, c1, {(loci, jj)})
        go = fill(go, c1, {(1, jj)})
    n2 = toindices(sfilter(dots, lambda cij: cij[0] == c2))
    n2coverage = apply(first, n2)
    if len(n2coverage) == h - 4 and h > 5:
        n2coverage = remove(choice(totuple(n2coverage)), n2coverage)
    for ii in n2coverage:
        locj = choice([ij[1] for ij in sfilter(n2, lambda ij: ij[0] == ii)])
        gi = fill(gi, c2, {(ii, locj)})
        go = fill(go, c2, {(ii, 1)})
    n3 = toindices(sfilter(dots, lambda cij: cij[0] == c4))
    n3coverage = apply(last, n3)
    if len(n3coverage) == w - 4 and w > 5:
        n3coverage = remove(choice(totuple(n3coverage)), n3coverage)
    for jj in n3coverage:
        loci = choice([ij[0] for ij in sfilter(n3, lambda ij: ij[1] == jj)])
        gi = fill(gi, c4, {(loci, jj)})
        go = fill(go, c4, {(h - 2, jj)})
    n4 = toindices(sfilter(dots, lambda cij: cij[0] == c3))
    n4coverage = apply(first, n4)
    if len(n4coverage) == h - 4 and h > 5:
        n4coverage = remove(choice(totuple(n4coverage)), n4coverage)
    for ii in n4coverage:
        locj = choice([ij[1] for ij in sfilter(n4, lambda ij: ij[0] == ii)])
        gi = fill(gi, c3, {(ii, locj)})
        go = fill(go, c3, {(ii, w - 2)})
    noisecands = ofcolor(gi, bgc)
    noisecols = difference(cols, (bgc, c1, c2, c3, c4))
    nnoise = unifint(diff_lb, diff_ub, (0, len(noisecands)))
    ub = ((h * w) - 2 * h - 2 * (w - 2)) // 2 - ndots - 1
    nnoise = unifint(diff_lb, diff_ub, (0, max(0, ub)))
    noise = sample(totuple(noisecands), nnoise)
    noiseobj = {(choice(noisecols), ij) for ij in noise}
    gi = paint(gi, noiseobj)
    return {'input': gi, 'output': go}


def generate_d90796e8(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8, 2, 3))
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc, noisec = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // 3))
    inds = asindices(gi)
    locs = sample(totuple(inds), nocc)
    obj = frozenset({(choice((noisec, 2, 3)), ij) for ij in locs})
    gi = paint(gi, obj)
    fixloc = choice(totuple(inds))
    fixloc2 = choice(totuple(dneighbors(fixloc) & inds))
    gi = fill(gi, 2, {fixloc})
    gi = fill(gi, 3, {fixloc2})
    go = tuple(e for e in gi)
    reds = ofcolor(gi, 2)
    greens = ofcolor(gi, 3)
    tocover = set()
    tolblue = set()
    for r in reds:
        inters = dneighbors(r) & greens
        if len(inters) > 0:
            tocover.add(r)
            tolblue = tolblue | inters
    go = fill(go, bgc, tocover)
    go = fill(go, 8, tolblue)
    return {'input': gi, 'output': go}


def generate_a68b268e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 14))
    w = unifint(diff_lb, diff_ub, (2, 4))
    bgc, linc, c1, c2, c3, c4 = sample(cols, 6)
    canv = canvas(bgc, (h, w))
    inds = asindices(canv)
    nc1d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc1 = choice((nc1d, h * w - nc1d))
    nc1 = min(max(1, nc1), h * w - 1)
    nc2d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc2 = choice((nc2d, h * w - nc2d))
    nc2 = min(max(1, nc2), h * w - 1)
    nc3d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc3 = choice((nc3d, h * w - nc3d))
    nc3 = min(max(1, nc3), h * w - 1)
    nc4d = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    nc4 = choice((nc4d, h * w - nc4d))
    nc4 = min(max(1, nc4), h * w - 1)
    ofc1 = sample(totuple(inds), nc1)
    ofc2 = sample(totuple(inds), nc2)
    ofc3 = sample(totuple(inds), nc3)
    ofc4 = sample(totuple(inds), nc4)
    go = fill(canv, c1, ofc1)
    go = fill(go, c2, ofc2)
    go = fill(go, c3, ofc3)
    go = fill(go, c4, ofc4)
    LR = asobject(fill(canv, c1, ofc1))
    LL = asobject(fill(canv, c2, ofc2))
    UR = asobject(fill(canv, c3, ofc3))
    UL = asobject(fill(canv, c4, ofc4))
    gi = canvas(linc, (2*h+1, 2*w+1))
    gi = paint(gi, shift(LR, (h+1, w+1)))
    gi = paint(gi, shift(LL, (h+1, 0)))
    gi = paint(gi, shift(UR, (0, w+1)))
    gi = paint(gi, shift(UL, (0, 0)))
    return {'input': gi, 'output': go}


def generate_ea32f347(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 4))
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    a = unifint(diff_lb, diff_ub, (3, 30))
    b = unifint(diff_lb, diff_ub, (2, a))
    c = unifint(diff_lb, diff_ub, (1, b))
    if c - a == 2:
        if a > 1:
            a -= 1
        elif c < min(h, w):
            c += 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    for col, l in zip((1, 4, 2), (a, b, c)):
        ln1 = connect((0, 0), (0, l - 1))
        ln2 = connect((0, 0), (l - 1, 0))
        tmpg = fill(gi, -1, asindices(gi) - inds)
        occs1 = occurrences(tmpg, recolor(bgc, ln1))
        occs2 = occurrences(tmpg, recolor(bgc, ln2))
        pool = []
        if len(occs1) > 0:
            pool.append((ln1, occs1))
        if len(occs2) > 0:
            pool.append((ln2, occs2))
        ln, occs = choice(pool)
        loc = choice(totuple(occs))
        plcd = shift(ln, loc)
        gi = fill(gi, choice(remcols), plcd)
        go = fill(go, col, plcd)
        inds = (inds - plcd) - mapply(dneighbors, plcd)
    return {'input': gi, 'output': go}


def generate_e179c5f4(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (2, 10))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    go = tuple(e for e in gi)
    changing = True
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        go2 = fill(go, linc, {sp})
        if go2 == go:
            break
        go = go2
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    gix = tuple(e for e in gi)
    gox = tuple(e for e in go)
    numlins = unifint(diff_lb, diff_ub, (1, 4))
    if numlins > 1:
        gi = fill(gi, linc, ofcolor(hmirror(gix), linc))
        go = fill(go, linc, ofcolor(hmirror(gox), linc))
    if numlins > 2:
        gi = fill(gi, linc, ofcolor(vmirror(gix), linc))
        go = fill(go, linc, ofcolor(vmirror(gox), linc))
    if numlins > 3:
        gi = fill(gi, linc, ofcolor(hmirror(vmirror(gix)), linc))
        go = fill(go, linc, ofcolor(hmirror(vmirror(gox)), linc))
    go = replace(go, bgc, 8)
    return {'input': gi, 'output': go}


def generate_aba27056(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    bgc, sqc = sample(cols, 2)
    canv = canvas(bgc, (h, w))
    oh = randint(3, h)
    ow = unifint(diff_lb, diff_ub, (5, w - 1))
    loci = unifint(diff_lb, diff_ub, (0, h - oh))
    locj = randint(0, w - ow)
    bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
    maxk = (ow - 4) // 2
    k = randint(0, maxk)
    hole = connect((loci, locj + 2 + k), (loci, locj + ow - 3 - k))
    gi = fill(canv, sqc, bx)
    gi = fill(gi, bgc, hole)
    go = fill(canv, 4, backdrop(bx))
    go = fill(go, sqc, bx)
    bar = mapply(rbind(shoot, (-1, 0)), hole)
    go = fill(go, 4, bar)
    go = fill(go, 4, shoot(add((-1, 1), urcorner(hole)), (-1, 1)))
    go = fill(go, 4, shoot(add((-1, -1), ulcorner(hole)), (-1, -1)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_e40b9e2f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    d = unifint(diff_lb, diff_ub, (4, min(h, w) - 2))
    loci = randint(0, h - d)
    locj = randint(0, w - d)
    loc = (loci, locj)
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    subg = canvas(bgc, (d, d))
    inds = asindices(subg)
    if d % 2 == 0:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] < d//2)
        cp = {(d//2-1, d//2-1), (d//2, d//2-1), (d//2-1, d//2), (d//2, d//2)}
    else:
        q = sfilter(inds, lambda ij: ij[0] < d//2 and ij[1] <= d//2)
        cp = {(d//2, d//2)} | ineighbors((d//2, d//2))
    nrings = unifint(diff_lb, diff_ub, (1, max(1, (d-2)//2)))
    rings = set()
    for k in range(nrings):
        ring = box({(k, k), (d-k-1, d-k-1)})
        rings = rings | ring
    qin = q - rings
    qout = rings & q
    ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
    tailobjcells = sample(totuple(q), ntailobjcells)
    tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
    tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    while hmirror(tailobj) == tailobj and vmirror(tailobj) == tailobj:
        ntailobjcells = unifint(diff_lb, diff_ub, (1, len(q)))
        tailobjcells = sample(totuple(q), ntailobjcells)
        tailobjcells = set(tailobjcells) | {choice(totuple(qin))} | {choice(totuple(qout))}
        tailobj = {(choice(ccols), ij) for ij in tailobjcells}
    for k in range(4):
        subg = paint(subg, tailobj)
        subg = rot90(subg)
    fxobj = recolor(choice(ccols), cp)
    subg = paint(subg, fxobj)
    subgi = subg
    subgo = tuple(e for e in subgi)
    subgi = fill(subgi, bgc, rings)
    nsplits = unifint(diff_lb, diff_ub, (1, 4))
    splits = [set() for k in range(nsplits)]
    for idx, cel in enumerate(tailobj):
        splits[idx%nsplits].add(cel)
    for jj in range(4):
        if jj < len(splits):
            subgi = paint(subgi, splits[jj])
        subgi = rot90(subgi)
    subgi = paint(subgi, fxobj)
    rotf = choice((identity, rot90, rot180, rot270))
    subgi = rotf(subgi)
    subgo = rotf(subgo)
    gi = paint(canvas(bgc, (h, w)), shift(asobject(subgi), loc))
    go = paint(canvas(bgc, (h, w)), shift(asobject(subgo), loc))
    return {'input': gi, 'output': go}


def generate_e8dc4411(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (9, 30))
    d = unifint(diff_lb, diff_ub, (3, min(h, w)//2-1))
    bgc, objc, remc = sample(cols, 3)
    c = canvas(bgc, (d, d))
    inds = sfilter(asindices(c), lambda ij: ij[0]>=d//2 and ij[1]>=d//2)
    ncd = unifint(diff_lb, diff_ub, (1, len(inds)//2))
    nc = choice((ncd, len(inds)-ncd))
    nc = min(max(2, nc), len(inds) - 1)
    cells = sample(totuple(inds), nc)
    cells = set(cells) | {choice(((d//2, d//2), (d//2, d//2-1)))}
    cells = cells | {(jj, ii) for ii, jj in cells}
    for k in range(4):
        c = fill(c, objc, cells)
        c = rot90(c)
    while palette(toobject(box(asindices(c)), c)) == frozenset({bgc}) and height(c) > 3:
        c = trim(c)
    obj = ofcolor(c, objc)
    od = height(obj)
    loci = randint(1, h - 2*od)
    locj = randint(1, w - 2*od)
    obj = shift(obj, (loci, locj))
    bd = backdrop(obj)
    p = 0
    while len(shift(obj, (p, p)) & bd) > 0:
        p += 1
    obj2 = shift(obj, (p, p))
    nbhs = mapply(neighbors, obj)
    while len(obj2 & nbhs) == 0:
        nbhs = mapply(neighbors, nbhs)
    indic = obj2 & nbhs
    gi = canvas(bgc, (h, w))
    gi = fill(gi, objc, obj)
    gi = fill(gi, remc, indic)
    go = tuple(e for e in gi)
    for k in range(30):
        newg = fill(go, remc, shift(obj, (p*(k+1), p*(k+1))))
        if newg == go:
            break
        go = newg
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_ddf7fa4f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)  
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nocc = unifint(diff_lb, diff_ub, (1, min(w // 3, (h * w) // 36)))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 10 * nocc
    inds = asindices(gi)
    inds = sfilter(inds, lambda ij: ij[0] > 1)
    while succ < nocc and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        hastobein = {cidx for cidx, col in enumerate(gi[0]) if col == bgc}
        cantbein = {cidx for cidx, col in enumerate(gi[0]) if col != bgc}
        jopts = [j for j in range(w) if \
            len(set(interval(j, j + ow, 1)) & hastobein) > 0 and len(set(interval(j, j + ow, 1)) & cantbein) == 0
        ]
        cands = sfilter(cands, lambda ij: ij[1] in jopts)
        if len(cands) == 0:
            continue
        loci, locj = choice(totuple(cands))
        locat = choice(sfilter(interval(locj, locj + ow, 1), lambda jj: jj in hastobein))
        sq = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if sq.issubset(inds):
            succ += 1
            inds = (inds - sq) - mapply(dneighbors, sq)
            col = choice(remcols)
            gr = choice(remove(col, remcols))
            gi = fill(gi, col, {(0, locat)})
            go = fill(go, col, {(0, locat)})
            gi = fill(gi, gr, sq)
            go = fill(go, col, sq)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_d07ae81c(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    lnf = lambda ij: shoot(ij, (1, 1)) | shoot(ij, (-1, -1)) | shoot(ij, (-1, 1)) | shoot(ij, (1, -1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    c1, c2, c3, c4 = sample(cols, 4)
    magiccol = 0
    gi = canvas(0, (h, w))
    ndivi = unifint(diff_lb, diff_ub, (1, (h * w) // 10))
    for k in range(ndivi):
        objs = objects(gi, T, F, F)
        objs = sfilter(objs, lambda o: min(shape(o)) > 3 and max(shape(o)) > 4)
        objs = sfilter(objs, lambda o: height(o) * width(o) == len(o))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        if choice((True, False)):
            loci = randint(uppermost(obj)+2, lowermost(obj)-1)
            newobj = backdrop(frozenset({(loci, leftmost(obj)), lrcorner(obj)}))
        else:
            locj = randint(leftmost(obj)+2, rightmost(obj)-1)
            newobj = backdrop(frozenset({(uppermost(obj), locj), lrcorner(obj)}))
        magiccol += 1
        gi = fill(gi, magiccol, newobj)
    objs = objects(gi, T, F, F)
    for ii, obj in enumerate(objs):
        col = c1 if ii == 0 else (c2 if ii == 1 else choice((c1, c2)))
        gi = fill(gi, col, toindices(obj))
    ofc1 = ofcolor(gi, c1)
    ofc2 = ofcolor(gi, c2)
    mn = min(len(ofc1), len(ofc2))
    n1 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    n2 = unifint(diff_lb, diff_ub, (1, max(1, int(mn ** 0.5))))
    srcs1 = set()
    for k in range(n1):
        cands = totuple((ofc1 - srcs1) - mapply(neighbors, srcs1))
        if len(cands) == 0:
            break
        srcs1.add(choice(cands))
    srcs2 = set()
    for k in range(n2):
        cands = totuple((ofc2 - srcs2) - mapply(neighbors, srcs2))
        if len(cands) == 0:
            break
        srcs2.add(choice(cands))
    gi = fill(gi, c3, srcs1)
    gi = fill(gi, c4, srcs2)
    lns = mapply(lnf, srcs1) | mapply(lnf, srcs2)
    ofc3 = ofc1 & lns
    ofc4 = ofc2 & lns
    go = fill(gi, c3, ofc3)
    go = fill(go, c4, ofc4)
    return {'input': gi, 'output': go}


def generate_b2862040(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (8,))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
        succ = 0
        tr = 0
        maxtr = 10 * nobjs
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        gi = canvas(bgc, (h, w))
        inds = asindices(gi)
        while succ < nobjs and tr < maxtr:
            tr += 1
            oh = randint(3, 6)
            ow = randint(3, 6)
            obj = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
            if choice((True, False)):
                nkeep = unifint(diff_lb, diff_ub, (2, len(obj) - 1))
                nrem = len(obj) - nkeep
                obj = remove(choice(totuple(obj - corners(obj))), obj)
                for k in range(nrem - 1):
                    xx = sfilter(obj, lambda ij: len(dneighbors(ij) & obj) == 1)
                    if len(xx) == 0:
                        break
                    obj = remove(choice(totuple(xx)), obj)
            npert = unifint(diff_lb, diff_ub, (0, oh + ow))
            objcands = outbox(obj) | outbox(outbox(obj)) | outbox(outbox(outbox(obj)))
            obj = set(obj)
            for k in range(npert):
                obj.add(choice(totuple((objcands - obj) & (mapply(dneighbors, obj) & objcands))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            plcd = shift(obj, loc)
            if plcd.issubset(inds):
                gi = fill(gi, choice(remcols), plcd)
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
        objs = objects(gi, T, F, F)
        bobjs = colorfilter(objs, bgc)
        objsm = mfilter(bobjs, compose(flip, rbind(bordering, gi)))
        if len(objsm) > 0:
            res = mfilter(objs - bobjs, rbind(adjacent, objsm))
            go = fill(gi, 8, res)
            break
    return {'input': gi, 'output': go}


def generate_a61ba2ce(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (4, 15))
    w = unifint(diff_lb, diff_ub, (4, 15))
    lociL = randint(2, h - 2)
    lociR = randint(2, h - 2)
    locjT = randint(2, w - 2)
    locjB = randint(2, w - 2)
    bgc, c1, c2, c3, c4 = sample(cols, 5)
    ulco = connect((0, 0), (lociL - 1, 0)) | connect((0, 0), (0, locjT - 1))
    urco = connect((0, w - 1), (0, locjT)) | connect((0, w - 1), (lociR - 1, w - 1))
    llco = connect((h - 1, 0), (lociL, 0)) | connect((h - 1, 0), (h - 1, locjB - 1))
    lrco = connect((h - 1, w - 1), (h - 1, locjB)) | connect((h - 1, w - 1), (lociR, w - 1))
    go = canvas(bgc, (h, w))
    go = fill(go, c1, ulco)
    go = fill(go, c2, urco)
    go = fill(go, c3, llco)
    go = fill(go, c4, lrco)
    fullh = unifint(diff_lb, diff_ub, (2 * h, 30))
    fullw = unifint(diff_lb, diff_ub, (2 * w, 30))
    gi = canvas(bgc, (fullh, fullw))
    objs = (ulco, urco, llco, lrco)
    ocols = (c1, c2, c3, c4)
    while True:
        inds = asindices(gi)
        locs = []
        for o, c in zip(objs, ocols):
            cands = sfilter(inds, lambda ij: shift(o, ij).issubset(inds))
            if len(cands) == 0:
                break
            loc = choice(totuple(cands))
            locs.append(loc)
            inds = inds - shift(o, loc)
        if len(locs) == 4:
            break
    for o, c, l in zip(objs, ocols, locs):
        gi = fill(gi, c, shift(o, l))
    return {'input': gi, 'output': go}


def generate_bbc9ae5d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    w = unifint(diff_lb, diff_ub, (2, 15))
    w = w * 2
    locinv = unifint(diff_lb, diff_ub, (2, w))
    locj = w - locinv
    loc = (0, locj)
    c1 = choice(cols)
    remcols = remove(c1, cols)
    ln1 = connect((0, 0), (0, locj))
    remobj = connect((0, locj+1), (0, w - 1))
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    remobj = {(choice(ccols), ij) for ij in remobj}
    gi = canvas(-1, (1, w))
    go = canvas(-1, (w//2, w))
    ln2 = shoot(loc, (1, 1))
    gi = fill(gi, c1, ln1)
    gi = paint(gi, remobj)
    go = fill(go, c1, mapply(rbind(shoot, (0, -1)), ln2))
    for c, ij in remobj:
        go = fill(go, c, shoot(ij, (1, 1)))
    return {'input': gi, 'output': go}


def generate_9edfc990(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(2, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    namt = unifint(diff_lb, diff_ub, (int(0.4 * h * w), int(0.7 * h * w)))
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    locs = sample(totuple(inds), namt)
    noise = {(choice(cols), ij) for ij in locs}
    gi = paint(gi, noise)
    remlocs = inds - set(locs)
    numc = unifint(diff_lb, diff_ub, (1, max(1, len(remlocs) // 10)))
    blocs = sample(totuple(remlocs), numc)
    gi = fill(gi, 1, blocs)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, 0)
    res = mfilter(objs, rbind(adjacent, blocs))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


def generate_a78176bb(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 30))
    w = unifint(diff_lb, diff_ub, (6, 30))
    nlns = unifint(diff_lb, diff_ub, (1, (h + w) // 8))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    succ = 0
    tr = 0
    maxtr = 10 * nlns
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))       
    inds = asindices(gi)
    fullinds = asindices(gi)
    spopts = []
    for idx in range(h - 5, -1, -1):
        spopts.append((idx, 0))
    for idx in range(1, w - 4, 1):
        spopts.append((0, idx))
    while succ < nlns and tr < maxtr:
        tr += 1
        if len(spopts) == 0:
            break
        sp = choice(spopts)
        ln = shoot(sp, (1, 1)) & fullinds
        if not ln.issubset(inds):
            continue
        lno = sorted(ln, key=lambda x: x[0])
        trid1 = randint(2, min(5, len(lno)-3))
        trid2 = randint(2, min(5, len(lno)-3))
        tri1 = sfilter(asindices(canvas(-1, (trid1, trid1))), lambda ij: ij[1] >= ij[0])
        triloc1 = add(choice(lno[1:-trid1-1]), (0, 1))
        tri2 = dmirror(sfilter(asindices(canvas(-1, (trid2, trid2))), lambda ij: ij[1] >= ij[0]))
        triloc2 = add(choice(lno[1:-trid2-1]), (1, 0))
        spo2 = add(sp, (0, -trid2-2))
        nexlin2 = {add(spo2, (k, k)) for k in range(max(h, w))} & fullinds
        spo1 = add(sp, (-trid1-2, 0))
        nexlin1 = {add(spo1, (k, k)) for k in range(max(h, w))} & fullinds
        for idx, (tri, triloc, nexlin) in enumerate(sample([
            (tri1, triloc1, nexlin1),
            (tri2, triloc2, nexlin2)
        ], 2)):
            tri = shift(tri, triloc)
            fullobj = ln | tri | nexlin
            if idx == 0:
                lncol, tricol = sample(remcols, 2)
            else:
                tricol = choice(remove(lncol, remcols))
            if (
                fullobj.issubset(inds) if idx == 0 else (tri | nexlin).issubset(fullobj)
            ):
                succ += 1
                inds = (inds - fullobj) - mapply(neighbors, fullobj)
                gi = fill(gi, tricol, tri)
                gi = fill(gi, lncol, ln)
                go = fill(go, lncol, ln)
                go = fill(go, lncol, nexlin)
    if choice((True, False)):
        gi = hmirror(gi)
        go = hmirror(go)
    return {'input': gi, 'output': go}


def generate_995c5fa3(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    o1 = asindices(canvas(-1, (4, 4)))
    o2 = box(asindices(canvas(-1, (4, 4))))
    o3 = asindices(canvas(-1, (4, 4))) - {(1, 0), (2, 0), (1, 3), (2, 3)}
    o4 = o1 - shift(asindices(canvas(-1, (2, 2))), (2, 1))
    mpr = [(o1, 2), (o2, 8), (o3, 3), (o4, 4)]
    num = unifint(diff_lb, diff_ub, (1, 6))
    h = 4
    w = 4 * num + num - 1
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    gi = canvas(bgc, (h, w))
    ccols = []
    for k in range(num):
        col = choice(remcols)
        obj, outcol = choice(mpr)
        locj = 5 * k
        gi = fill(gi, col, shift(obj, (0, locj)))
        ccols.append(outcol)
    go = tuple(repeat(c, num) for c in ccols)
    return {'input': gi, 'output': go}


def generate_9aec4887(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    oh = unifint(diff_lb, diff_ub, (4, h//2-2))
    ow = unifint(diff_lb, diff_ub, (4, w//2-2))
    bgc, pc, c1, c2, c3, c4 = sample(cols, 6)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    ln3 = connect((0, 1), (0, ow - 2))
    ln4 = connect((oh - 1, 1), (oh - 1, ow - 2))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, c3, ln3)
    go = fill(go, c4, ln4)
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    gi = paint(gi, plcdB)
    gi = fill(gi, pc, plcdA)
    objA = shift(objA, (1, 1))
    objs = objects(go, T, F, T)
    for ij in objA:
        manhs = {obj: manhattan(obj, {ij}) for obj in objs}
        manhsl = list(manhs.values())
        mmh = min(manhsl)
        if manhsl.count(mmh) == 1:
            col = color([o for o, mnh in manhs.items() if mmh == mnh][0])
        else:
            col = pc
        go = fill(go, col, {ij})
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_846bdb03(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    oh = unifint(diff_lb, diff_ub, (4, h//2-2))
    ow = unifint(diff_lb, diff_ub, (4, w//2-2))
    bgc, dotc, c1, c2 = sample(cols, 4)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (oh, ow))
    ln1 = connect((1, 0), (oh - 2, 0))
    ln2 = connect((1, ow - 1), (oh - 2, ow - 1))
    go = fill(go, c1, ln1)
    go = fill(go, c2, ln2)
    go = fill(go, dotc, corners(asindices(go)))
    objB = asobject(go)
    bounds = asindices(canvas(-1, (oh - 2, ow - 2)))
    objA = {choice(totuple(bounds))}
    ncells = unifint(diff_lb, diff_ub, (1, ((oh - 2) * (ow - 2)) // 2))
    for k in range(ncells - 1):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    while shape(objA) != (oh - 2, ow - 2):
        objA.add(choice(totuple((bounds - objA) & mapply(neighbors, objA))))
    fullinds = asindices(gi)
    loci = randint(0, h - 2 * oh + 2)
    locj = randint(0, w - ow)
    plcdB = shift(objB, (loci, locj))
    plcdi = toindices(plcdB)
    rems = sfilter(fullinds - plcdi, lambda ij: loci + oh <= ij[0] <= h - oh + 2 and ij[1] <= w - ow + 2)
    loc = choice(totuple(rems))
    plcdA = shift(objA, loc)
    mp = center(plcdA)[1]
    plcdAL = sfilter(plcdA, lambda ij: ij[1] < mp)
    plcdAR = plcdA - plcdAL
    plcdA = recolor(c1, plcdAL) | recolor(c2, plcdAR)
    gi = paint(gi, plcdB)
    ism = choice((True, False))
    gi = paint(gi, vmirror(plcdA) if ism else plcdA)
    objA = shift(normalize(plcdA), (1, 1))
    objs = objects(go, T, F, T)
    go = paint(go, objA)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_2dd70a9a(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    if choice((True, False)):
        oh = unifint(diff_lb, diff_ub, (5, h - 2))
        ow = unifint(diff_lb, diff_ub, (3, w - 2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        hli = randint(loci+2, loci+oh-3)
        sp = {(loci+oh-1, locj), (loci+oh-2, locj)}
        ep = {(loci, locj+ow-1), (loci+1, locj+ow-1)}
        bp1 = (hli-1, locj)
        bp2 = (hli, locj+ow)
        ln1 = connect((loci+oh-1, locj), (hli, locj))
        ln2 = connect((hli, locj), (hli, locj+ow-1))
        ln3 = connect((hli, locj+ow-1), (loci+2, locj+ow-1))
    else:
        oh = unifint(diff_lb, diff_ub, (3, h-2))
        ow = unifint(diff_lb, diff_ub, (3, w-2))
        loci = randint(1, h - oh - 1)
        locj = randint(1, w - ow - 1)
        if choice((True, False)):
            sp1j = randint(locj, locj+ow-3)
            ep1j = locj
        else:
            ep1j = randint(locj, locj+ow-3)
            sp1j = locj
        sp = {(loci, sp1j), (loci, sp1j+1)}
        ep = {(loci+oh-1, ep1j), (loci+oh-1, ep1j+1)}
        bp1 = (loci, locj+ow)
        bp2 = (loci+oh, locj+ow-1)
        ln1 = connect((loci, sp1j+2), (loci, locj+ow-1))
        ln2 = connect((loci, locj+ow-1), (loci+oh-1, locj+ow-1))
        ln3 = connect((loci+oh-1, ep1j+2), (loci+oh-1, locj+ow-1))
    gi = fill(gi, 3, sp)
    gi = fill(gi, 2, ep)
    go = fill(go, 3, sp)
    go = fill(go, 2, ep)
    lns = ln1 | ln2 | ln3
    bps = {bp1, bp2}
    gi = fill(gi, fgc, bps)
    go = fill(go, fgc, bps)
    go = fill(go, 3, lns)
    inds = ofcolor(go, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(inds) // 2))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = fill(go, fgc, noise)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_36fdfd69(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (4,))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nobjs = unifint(diff_lb, diff_ub, (1, (h * w) // 30))
    bgc, fgc, objc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    namt = randint(int(0.35 * h * w), int(0.65 * h * w))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, fgc, noise)
    go = tuple(e for e in gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        oh = randint(2, 7)
        ow = randint(2, 7)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        if bd.issubset(inds):
            ncells = randint(2, oh * ow - 1)
            obj = {choice(totuple(bd))}
            for k in range(ncells - 1):
                obj.add(choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            while len(obj) == height(obj) * width(obj):
                obj = {choice(totuple(bd))}
                for k in range(ncells - 1):
                    obj.add(choice(totuple((bd - obj) & mapply(neighbors, mapply(dneighbors, obj)))))
            obj = normalize(obj)
            oh, ow = shape(obj)
            obj = shift(obj, loc)
            bd = backdrop(obj)
            gi2 = fill(gi, fgc, bd)
            gi2 = fill(gi2, objc, obj)
            if colorcount(gi2, objc) < min(colorcount(gi2, fgc), colorcount(gi2, bgc)):
                succ += 1
                inds = (inds - bd) - (outbox(bd) | outbox(outbox(bd)))
                gi = gi2
                go = fill(go, 4, bd)
                go = fill(go, objc, obj)
    return {'input': gi, 'output': go}


def generate_28e73c20(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (3,))
    direcmapper = {(0, 1): (1, 0), (1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (4, 30))
    sp = (0, w - 1)
    direc = (1, 0)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    inds = asindices(gi)
    obj = {(choice(ccols), ij) for ij in inds}
    gi = paint(gi, obj)
    go = fill(gi, 3, connect((0, 0), sp))
    lw = w
    lh = h
    ld = h
    isverti = False
    while ld > 0:
        lw -= 1
        lh -= 1
        ep = add(sp, multiply(direc, ld - 1))
        ln = connect(sp, ep)
        go = fill(go, 3, ln)
        direc = direcmapper[direc]
        if isverti:
            ld = lh
        else:
            ld = lw
        isverti = not isverti
        sp = ep
    gi = dmirror(dmirror(gi)[1:])
    go = dmirror(dmirror(go)[1:])
    return {'input': gi, 'output': go}


def generate_3eda0437(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(1, 10, 1), (6,))
    h = unifint(diff_lb, diff_ub, (3, 8))
    w = unifint(diff_lb, diff_ub, (3, 30))
    if choice((True, False)):
        h, w = w, h
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    fgcs = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    gi = paint(gi, {(choice(fgcs), ij) for ij in asindices(gi)})
    spac = unifint(diff_lb, diff_ub, (1, (h * w) // 3 * 2))
    inds = asindices(gi)
    obj = sample(totuple(inds), spac)
    gi = fill(gi, 0, obj)
    locx = (randint(0, h-1), randint(0, w-1))
    gi = fill(gi, 0, {locx, add(locx, RIGHT), add(locx, DOWN), add(locx, UNITY)})
    maxsiz = -1
    mapper = dict()
    maxpossw = max([r.count(0) for r in gi])
    maxpossh = max([c.count(0) for c in dmirror(gi)])
    for a in range(2, maxpossh+1):
        for b in range(2, maxpossw+1):
            siz = a * b
            if siz < maxsiz:
                continue
            objx = recolor(0, asindices(canvas(-1, (a, b))))
            occs = occurrences(gi, objx)
            if len(occs) > 0:
                if siz == maxsiz:
                    mapper[objx] = occs
                elif siz > maxsiz:
                    mapper = {objx: occs}
                    maxsiz = siz
    go = tuple(e for e in gi)
    for obj, locs in mapper.items():
        go = fill(go, 6, mapply(lbind(shift, obj), locs))
    return {'input': gi, 'output': go}


def generate_7447852a(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    w = unifint(diff_lb, diff_ub, (2, 8))
    h = unifint(diff_lb, diff_ub, (w+1, 30))
    bgc, linc = sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    c = canvas(bgc, (h, w))
    sp = (h - 1, 0)
    gi = fill(c, linc, {sp})
    direc = 1
    while True:
        sp = add(sp, (-1, direc))
        if sp[1] == w - 1 or sp[1] == 0:
            direc *= -1
        gi2 = fill(gi, linc, {sp})
        if gi2 == gi:
            break
        gi = gi2
    gi = rot90(gi)
    objs = objects(gi, T, F, F)
    inds = ofcolor(gi, bgc)
    numcols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numcols)
    ncells = unifint(diff_lb, diff_ub, (0, len(inds)))
    locs = sample(totuple(inds), ncells)
    obj = {(choice(ccols), ij) for ij in locs}
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    objs = order(colorfilter(objs, bgc), leftmost)
    objs = merge(set(objs[0::3]))
    go = fill(go, 4, objs)
    return {'input': gi, 'output': go}


def generate_6b9890af(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (2, 5))
    ow = unifint(diff_lb, diff_ub, (2, 5))
    h = unifint(diff_lb, diff_ub, (2*oh+2, 30))
    w = unifint(diff_lb, diff_ub, (2*ow+2, 30))
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    while shape(obj) != (oh, ow):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    maxfac = 1
    while oh * maxfac + 2 <= h - oh and ow * maxfac + 2 <= w - ow:
        maxfac += 1
    maxfac -= 1
    fac = unifint(diff_lb, diff_ub, (1, maxfac))
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    obj = {(choice(ccols), ij) for ij in obj}
    gi = canvas(bgc, (h, w))
    sq = box(frozenset({(0, 0), (oh * fac + 1, ow * fac + 1)}))
    loci = randint(0, h - (oh * fac + 2) - oh)
    locj = randint(0, w - (ow * fac + 2))
    gi = fill(gi, sqc, shift(sq, (loci, locj)))
    loci = randint(loci+oh*fac+2, h - oh)
    locj = randint(0, w - ow)
    objp = shift(obj, (loci, locj))
    gi = paint(gi, objp)
    go = canvas(bgc, (oh * fac + 2, ow * fac + 2))
    go = fill(go, sqc, sq)
    go2 = paint(canvas(bgc, (oh, ow)), obj)
    upscobj = asobject(upscale(go2, fac))
    go = paint(go, shift(upscobj, (1, 1)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_963e52fc(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (6, 15))
    p = unifint(diff_lb, diff_ub, (2, w // 2))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numc)
    obj = set()
    for j in range(p):
        ub = unifint(diff_lb, diff_ub, (0, h//2))
        ub = h//2-ub
        lb = unifint(diff_lb, diff_ub, (ub, h-1))
        numcells = unifint(diff_lb, diff_ub, (1, lb-ub+1))
        for ii in sample(interval(ub, lb+1, 1), numcells):
            loc = (ii, j)
            col = choice(ccols)
            cell = (col, loc)
            obj.add(cell)
    go = canvas(bgc, (h, w*2))
    minobj = obj | shift(obj, (0, p))
    addonw = randint(0, p)
    addon = sfilter(obj, lambda cij: cij[1][1] < addonw)
    fullobj = minobj | addon
    leftshift = randint(0, addonw)
    fullobj = shift(fullobj, (0, -leftshift))
    go = paint(go, fullobj)
    for j in range((2*w)//(2*p)+1):
        go = paint(go, shift(fullobj, (0, j * 2 * p)))
    gi = lefthalf(go)
    return {'input': gi, 'output': go}


def generate_3e980e27(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (2, 3))
    h = unifint(diff_lb, diff_ub, (11, 30))
    w = unifint(diff_lb, diff_ub, (11, 30))
    bgc, rcol, gcol = sample(cols, 3)
    objs = []
    for (fixc, remc) in ((2, rcol), (3, gcol)):
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (2, 5))
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(2, ncells), oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        fixp = choice(totuple(obj))
        rem = remove(fixp, obj)
        obj = {(fixc, fixp)} | recolor(remc, rem)
        objs.append(obj)
    robj, gobj = objs
    obj1, obj2 = sample(objs, 2)
    loci1 = randint(0, h - height(obj1) - height(obj2) - 1)
    locj1 = randint(0, w - width(obj1))
    loci2 = randint(loci1+height(obj1)+1, h - height(obj2))
    locj2 = randint(0, w - width(obj2))
    gi = canvas(bgc, (h, w))
    obj1p = shift(obj1, (loci1, locj1))
    obj2p = shift(obj2, (loci2, locj2))
    gi = paint(gi, obj1p)
    gi = paint(gi, obj2p)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // int(1.5 * (len(robj) + len(gobj)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    robj = vmirror(robj)
    inds = ofcolor(gi, bgc) - (mapply(neighbors, toindices(obj1p)) | mapply(neighbors, toindices(obj2p)))
    go = tuple(e for e in gi)
    objopts = [robj, gobj]
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = choice(objopts)
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] in (2, 3)))
            go = paint(go, plcd)
    if unifint(diff_lb, diff_ub, (1, 100)) < 30:
        c = choice((2, 3))
        giobjs = objects(gi, F, T, T)
        goobjs = objects(go, F, T, T)
        gi = fill(gi, bgc, mfilter(giobjs, lambda o: c in palette(o)))
        go = fill(go, bgc, mfilter(goobjs, lambda o: c in palette(o)))
    return {'input': gi, 'output': go}


def generate_a8c38be5(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    goh = unifint(diff_lb, diff_ub, (9, 20))
    gow = unifint(diff_lb, diff_ub, (9, 20))
    h = unifint(diff_lb, diff_ub, (goh+4, 30))
    w = unifint(diff_lb, diff_ub, (gow+4, 30))
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    numc = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numc)
    go = canvas(sqc, (goh, gow))
    go = fill(go, bgc, box(asindices(go)))
    loci1 = randint(2, goh-7)
    loci2 = randint(loci1+4, goh-3)
    locj1 = randint(2, gow-7)
    locj2 = randint(locj1+4, gow-3)
    f1 = hfrontier((loci1, 0))
    f2 = hfrontier((loci2, 0))
    f3 = vfrontier((0, locj1))
    f4 = vfrontier((0, locj2))
    fs = f1 | f2 | f3 | f4
    go = fill(go, sqc, fs)
    go = fill(go, bgc, {((loci1 + loci2) // 2, 1)})
    go = fill(go, bgc, {((loci1 + loci2) // 2, gow - 2)})
    go = fill(go, bgc, {(1, (locj1 + locj2) // 2)})
    go = fill(go, bgc, {(goh - 2, (locj1 + locj2) // 2)})
    objs = objects(go, T, F, T)
    objs = merge(set(recolor(choice(ccols), obj) for obj in objs))
    go = paint(go, objs)
    gi = go
    hdelt = h - goh
    hdelt1 = randint(1, hdelt - 3)
    hdelt2 = randint(1, hdelt - hdelt1 - 2)
    hdelt3 = randint(1, hdelt - hdelt1 - hdelt2 - 1)
    hdelt4 = hdelt - hdelt1 - hdelt2 - hdelt3
    wdelt = w - gow
    wdelt1 = randint(1, wdelt - 3)
    wdelt2 = randint(1, wdelt - wdelt1 - 2)
    wdelt3 = randint(1, wdelt - wdelt1 - wdelt2 - 1)
    wdelt4 = wdelt - wdelt1 - wdelt2 - wdelt3
    gi = gi[:loci2] + repeat(repeat(bgc, gow), hdelt2) + gi[loci2:]
    gi = gi[:loci1+1] + repeat(repeat(bgc, gow), hdelt3) + gi[loci1+1:]
    gi = repeat(repeat(bgc, gow), hdelt1) + gi + repeat(repeat(bgc, gow), hdelt4)
    gi = dmirror(gi)
    gi = gi[:locj2] + repeat(repeat(bgc, h), wdelt2) + gi[locj2:]
    gi = gi[:locj1+1] + repeat(repeat(bgc, h), wdelt3) + gi[locj1+1:]
    gi = repeat(repeat(bgc, h), wdelt1) + gi + repeat(repeat(bgc, h), wdelt4)
    gi = dmirror(gi)
    nswitcheroos = unifint(diff_lb, diff_ub, (0, 10))
    if choice((True, False)):
        gi = gi[loci1+hdelt1+1:] + gi[:loci1+hdelt1+1]
    if choice((True, False)):
        gi = dmirror(gi)
        gi = gi[locj1+wdelt1+1:] + gi[:locj1+wdelt1+1]
        gi = dmirror(gi)
    for k in range(nswitcheroos):
        o = asobject(gi)
        tmpc = canvas(bgc, (h+12, w+12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, T, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        mpr = dict()
        for obj in objs:
            shp = shape(obj)
            if shp in mpr:
                mpr[shp].append(obj)
            else:
                mpr[shp] = [obj]
        if max([len(x) for x in mpr.values()]) == 1:
            break
        ress = [(kk, v) for kk, v in mpr.items() if len(v) > 1]
        res, abc = choice(ress)
        a, b = sample(abc, 2)
        ulca = ulcorner(a)
        ulcb = ulcorner(b)
        ap = shift(normalize(a), ulcb)
        bp = shift(normalize(b), ulca)
        gi = paint(gi, ap | bp)
    nshifts = unifint(diff_lb, diff_ub, (0, 30))
    for k in range(nshifts):
        o = asobject(gi)
        tmpc = canvas(bgc, (h+12, w+12))
        tmpc = paint(tmpc, shift(o, (6, 6)))
        objs = objects(tmpc, F, F, T)
        objs = apply(rbind(shift, (-6, -6)), objs)
        objs = sfilter(objs, compose(flip, rbind(bordering, gi)))
        if len(objs) == 0:
            break
        obj = choice(totuple(objs))
        direc1 = (randint(-1, 1), randint(-1, 1))
        direc2 = position({(h//2, w//2)}, {center(obj)})
        direc = choice((direc1, direc2))
        gi = fill(gi, bgc, obj)
        gi = paint(gi, shift(obj, direc))
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_6c434453(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    nobjs = unifint(diff_lb, diff_ub, (2, (h * w) // 16))
    succ = 0
    tr = 0
    maxtr = 5 * nobjs
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    while succ < nobjs and tr < maxtr:
        tr += 1
        if choice((True, False)):
            oh = choice((3, 5))
            ow = choice((3, 5))
            obji = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        else:
            oh = randint(1, 5)
            ow = randint(1, 5)
            bounds = asindices(canvas(-1, (oh, ow)))
            ncells = randint(1, oh * ow)
            obji = {choice(totuple(bounds))}
            for k in range(ncells - 1):
                obji.add(choice(totuple((bounds - obji) & mapply(dneighbors, obji))))
            obji = normalize(obji)
        oh, ow = shape(obji)
        flag = obji == box(obji) and set(shape(obji)).issubset({3, 5})
        if flag:
            objo = connect((0, ow//2), (oh - 1, ow//2)) | connect((oh//2, 0), (oh//2, ow - 1))
            tocover = backdrop(obji)
        else:
            objo = obji
            tocover = obji
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        loc = choice(totuple(cands))
        plcdi = shift(obji, loc)
        if plcdi.issubset(inds):
            plcdo = shift(objo, loc)
            succ += 1
            tocoveri = shift(tocover, loc)
            inds = (inds - tocoveri) - mapply(dneighbors, tocoveri)
            col = choice(ccols)
            gi = fill(gi, col, plcdi)
            go = fill(go, 2 if flag else col, plcdo)
    return {'input': gi, 'output': go}


def generate_7837ac64(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bgc, linc = sample(cols, 2)
    remcols = remove(bgc, remove(linc, cols))
    numcols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, numcols)
    go = canvas(bgc, (oh, ow))
    inds = asindices(go)
    fullinds = asindices(go)
    nocc = unifint(diff_lb, diff_ub, (1, oh * ow))
    for k in range(nocc):
        mpr = {
            cc: sfilter(
                inds | mapply(neighbors, ofcolor(go, cc)),
                lambda ij: (neighbors(ij) & fullinds).issubset(inds | ofcolor(go, cc))
            ) for cc in ccols
        }
        mpr = [(kk, vv) for kk, vv in mpr.items() if len(vv) > 0]
        if len(mpr) == 0:
            break
        col, locs = choice(mpr)
        loc = choice(totuple(locs))
        go = fill(go, col, {loc})
        inds = remove(loc, inds)
    obj = fullinds - ofcolor(go, bgc)
    go = subgrid(obj, go)
    oh, ow = shape(go)
    sqsizh = unifint(diff_lb, diff_ub, (2, (30 - oh - 1) // oh))
    sqsizw = unifint(diff_lb, diff_ub, (2, (30 - ow - 1) // ow))
    fullh = oh + 1 + oh * sqsizh
    fullw = ow + 1 + ow * sqsizw
    gi = canvas(linc, (fullh, fullw))
    sq = backdrop(frozenset({(0, 0), (sqsizh - 1, sqsizw - 1)}))
    obj = asobject(go)
    for col, ij in obj:
        plcd = shift(sq, add((1, 1), multiply(ij, (sqsizh+1, sqsizw+1))))
        gi = fill(gi, bgc, plcd)
        if col != bgc:
            gi = fill(gi, col, corners(outbox(plcd)))
    gih = unifint(diff_lb, diff_ub, (fullh, 30))
    giw = unifint(diff_lb, diff_ub, (fullw, 30))
    loci = randint(0, gih - fullh)
    locj = randint(0, giw - fullw)
    gigi = canvas(bgc, (gih, giw))
    plcd = shift(asobject(gi), (loci, locj))
    gigi = paint(gigi, plcd)
    ulci, ulcj = ulcorner(plcd)
    lrci, lrcj = lrcorner(plcd)
    for a in range(ulci, gih+1, sqsizh+1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for a in range(ulci, -1, -sqsizh-1):
        gigi = fill(gigi, linc, hfrontier((a, 0)))
    for b in range(ulcj, giw+1, sqsizw+1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    for b in range(ulcj, -1, -sqsizw-1):
        gigi = fill(gigi, linc, vfrontier((0, b)))
    gi = paint(gigi, plcd)
    gop = palette(go)
    while True:
        go2 = identity(go)
        for c in set(ccols) & gop:
            o1 = frozenset({(c, ORIGIN), (bgc, RIGHT), (c, (0, 2))})
            o2 = dmirror(o1)
            go2 = fill(go2, c, combine(
                shift(occurrences(go, o1), RIGHT),
                shift(occurrences(go, o2), DOWN)
            ))
        if go2 == go:
            break
        go = go2
    return {'input': gi, 'output': go}


def generate_5ad4f10b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nbh = {(0, 0), (1, 0), (0, 1), (1, 1)}
    nbhs = apply(lbind(shift, nbh), {(0, 0), (-1, 0), (0, -1), (-1, -1)})
    oh = unifint(diff_lb, diff_ub, (2, 6))
    ow = unifint(diff_lb, diff_ub, (2, 6))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow - 1)
    obj = set(sample(totuple(bounds), ncells))
    while len(sfilter(obj, lambda ij: sum([len(obj & shift(nbh, ij)) < 4 for nbh in nbhs]) > 0)) == 0:
        ncellsd = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        ncells = choice((ncellsd, oh * ow - ncellsd))
        ncells = min(max(1, ncells), oh * ow)
        obj = set(sample(totuple(bounds), ncells))
    obj = normalize(obj)
    oh, ow = shape(obj)
    bgc, noisec, objc = sample(cols, 3)
    go = canvas(bgc, (oh, ow))
    go = fill(go, noisec, obj)
    fac = unifint(diff_lb, diff_ub, (2, min(28//oh, 28//ow)))
    gobj = asobject(upscale(replace(go, noisec, objc), fac))
    oh, ow = shape(gobj)
    h = unifint(diff_lb, diff_ub, (oh+2, 30))
    w = unifint(diff_lb, diff_ub, (ow+2, 30))
    loci = randint(1, h - oh-1)
    locj = randint(1, w - ow-1)
    gi = canvas(bgc, (h, w))
    gi = paint(gi, shift(gobj, (loci, locj)))
    cands = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (2, max(1, len(cands) // 4)))
    noise = sample(totuple(cands), namt)
    gi = fill(gi, noisec, noise)
    return {'input': gi, 'output': go}


def generate_7df24a62(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 32))
    w = unifint(diff_lb, diff_ub, (12, 32))
    oh = unifint(diff_lb, diff_ub, (3, min(7, h//3)))
    ow = unifint(diff_lb, diff_ub, (3, min(7, w//3)))
    bgc, noisec, sqc = sample(cols, 3)
    tmpg = canvas(sqc, (oh, ow))
    inbounds = backdrop(inbox(asindices(tmpg)))
    obj = {choice(totuple(inbounds))}
    while shape(obj) != (oh - 2, ow - 2):
        obj.add(choice(totuple(inbounds - obj)))
    pat = fill(tmpg, noisec, obj)
    targ = asobject(fill(canvas(bgc, (oh, ow)), noisec, obj))
    sour = asobject(pat)
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - oh - 1)
    locj = randint(1, w - ow - 1)
    plcddd = shift(sour, (loci, locj))
    gi = paint(gi, plcddd)
    inds = ofcolor(gi, bgc) & shift(asindices(canvas(-1, (h-2, w-2))), (1, 1))
    inds = inds - (toindices(plcddd) | mapply(dneighbors, toindices(plcddd)))
    namt = unifint(diff_lb, diff_ub, (1, max(1, len(inds) // 4)))
    noise = sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    targs = []
    sours = []
    for fn1 in (identity, dmirror, cmirror, hmirror, vmirror):
        for fn2 in (identity, dmirror, cmirror, hmirror, vmirror):
            targs.append(normalize(fn1(fn2(targ))))
            sours.append(normalize(fn1(fn2(sour))))
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // ((oh * ow * 4)))))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        t = choice(targs)
        hh, ww = shape(t)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - hh - 1 and 1 <= ij[1] <= w - ww - 1)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        tp = shift(t, loc)
        tpi = toindices(tp)
        if tpi.issubset(inds):
            succ += 1
            inds = inds - tpi
            gi = paint(gi, tp)
    go = replace(gi, sqc, bgc)
    go = paint(go, plcddd)
    res = set()
    for t, s in zip(targs, sours):
        res |= mapply(lbind(shift, s), occurrences(go, t))
    go = paint(go, res)
    gi = trim(gi)
    go = trim(go)
    return {'input': gi, 'output': go}


def generate_539a4f51(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    d = unifint(diff_lb, diff_ub, (2, 15))
    h, w = d, d
    gi = canvas(0, (h, w))
    numc = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(cols, numc)
    numocc = unifint(diff_lb, diff_ub, (1, d))
    arr = [choice(ccols) for k in range(numocc)]
    while len(set(arr)) == 1:
        arr = [choice(ccols) for k in range(d)]
    for j, col in enumerate(arr):
        gi = fill(gi, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    go = canvas(0, (2*d, 2*d))
    for j in range(2*d):
        col = arr[j % len(arr)]
        go = fill(go, col, connect((j, 0), (j, j)) | connect((0, j), (j, j)))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_ce602527(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    bgc, c1, c2, c3 = sample(cols, 4)
    while True:
        objs = []
        for k in range(2):
            oh1 = unifint(diff_lb, diff_ub, (3, h//3-1))
            ow1 = unifint(diff_lb, diff_ub, (3, w//3-1))
            cc1 = canvas(bgc, (oh1, ow1))
            bounds1 = asindices(cc1)
            numcd1 = unifint(diff_lb, diff_ub, (0, (oh1 * ow1) // 2 - 4))
            numc1 = choice((numcd1, oh1 * ow1 - numcd1))
            numc1 = min(max(3, numc1), oh1 * ow1 - 3)
            obj1 = {choice(totuple(bounds1))}
            while len(obj1) < numc1 or shape(obj1) != (oh1, ow1):
                obj1.add(choice(totuple((bounds1 - obj1) & mapply(dneighbors, obj1))))
            objs.append(normalize(obj1))
        a, b = objs
        ag = fill(canvas(0, shape(a)), 1, a)
        bg = fill(canvas(0, shape(b)), 1, b)
        maxinh = min(height(a), height(b)) // 2 + 1
        maxinw = min(width(a), width(b)) // 2 + 1
        maxshp = (maxinh, maxinw)
        ag = crop(ag, (0, 0), maxshp)
        bg = crop(bg, (0, 0), maxshp)
        if ag != bg:
            break
    a, b = objs
    trgo = choice(objs)
    trgo2 = ofcolor(upscale(fill(canvas(0, shape(trgo)), 1, trgo), 2), 1)
    staysinh = unifint(diff_lb, diff_ub, (maxinh * 2, height(trgo) * 2))
    nout = height(trgo2) - staysinh
    loci = h - height(trgo2) + nout
    locj = randint(0, w - maxinw * 2)
    gi = canvas(bgc, (h, w))
    gi = fill(gi, c3, shift(trgo2, (loci, locj)))
    (lcol, lobj), (rcol, robj) = sample([(c1, a), (c2, b)], 2)
    cands = ofcolor(gi, bgc) - box(asindices(gi))
    lca = sfilter(cands, lambda ij: ij[1] < w//3*2)
    rca = sfilter(cands, lambda ij: ij[1] > w//3)
    lcands = sfilter(lca, lambda ij: shift(lobj, ij).issubset(lca))
    rcands = sfilter(rca, lambda ij: shift(robj, ij).issubset(rca))
    while True:
        lloc = choice(totuple(lcands))
        rloc = choice(totuple(lcands))
        lplcd = shift(lobj, lloc)
        rplcd = shift(robj, rloc)
        if lplcd.issubset(cands) and rplcd.issubset(cands) and len(lplcd & rplcd) == 0:
            break
    gi = fill(gi, lcol, shift(lobj, lloc))
    gi = fill(gi, rcol, shift(robj, rloc))
    go = fill(canvas(bgc, shape(trgo)), c1 if trgo == a else c2, trgo)
    mfs = (identity, rot90, rot180, rot270, cmirror, dmirror, hmirror, vmirror)
    mf = choice(mfs)
    gi, go = mf(gi), mf(go)
    return {'input': gi, 'output': go}


def generate_c8cbb738(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    gh = unifint(diff_lb, diff_ub, (3, 10))
    gw = unifint(diff_lb, diff_ub, (3, 10))
    h = unifint(diff_lb, diff_ub, (gh*2, 30))
    w = unifint(diff_lb, diff_ub, (gw*2, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (gh, gw))
    goinds = asindices(go)
    ring = box(goinds)
    crns = corners(ring)
    remring = ring - crns
    nrr = len(remring)
    sc = ccols[0]
    go = fill(go, sc, crns)
    loci = randint(0, h - gh)
    locj = randint(0, w - gw)
    gi = fill(gi, sc, shift(crns, (loci, locj)))
    ccols = ccols[1:]
    issucc = True
    bL = connect((0, 0), (gh - 1, 0))
    bR = connect((0, gw - 1), (gh - 1, gw - 1))
    bT = connect((0, 0), (0, gw - 1))
    bB = connect((gh - 1, 0), (gh - 1, gw - 1))
    validpairs = [(bL, bT), (bL, bB), (bR, bT), (bR, bB)]
    for c in ccols:
        if len(remring) < 3:
            break
        obj = set(sample(totuple(remring), unifint(diff_lb, diff_ub, (3, max(3, min(len(remring), nrr//len(ccols)))))))
        flag = False
        for b1, b2 in validpairs:
            if len(obj & b1) > 0 and len(obj & b2) > 0:
                flag = True
                break
        if flag:
            oh, ow = shape(obj)
            locs = ofcolor(gi, bgc)
            cands = sfilter(locs, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) > 0:
                objn = normalize(obj)
                cands2 = sfilter(cands, lambda ij: shift(objn, ij).issubset(locs))
                if len(cands2) > 0:
                    loc = choice(totuple(cands2))
                    gi = fill(gi, c, shift(objn, loc))
                    go = fill(go, c, obj)
                    remring -= obj
    return {'input': gi, 'output': go}


def generate_b527c5c6(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (2, 9))
    ccols = sample(remcols, ncols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, 10))
    tr = 0
    succ = 0
    maxtr = 10 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        d1 = randint(3, randint(3, (min(h, w)) // 2 - 1))
        d2 = randint(d1*2+1, randint(d1*2+1, min(h, w) - 1))
        oh, ow = sample([d1, d2], 2)
        cands = sfilter(inds, lambda ij: 1 <= ij[0] <= h - oh - 1 and 1 <= ij[1] <= w - ow - 1)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bx = box(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        bd = backdrop(bx)
        if ow < oh:
            lrflag = True
            dcands1 = connect((loci+ow-1, locj), (loci+oh-1-ow+1, locj))
            dcands2 = shift(dcands1, (0, ow-1))
        else:
            lrflag = False
            dcands1 = connect((loci, locj+oh-1), (loci, locj+ow-1-oh+1))
            dcands2 = shift(dcands1, (oh-1, 0))
        dcands = dcands1 | dcands2
        loc = choice(totuple(dcands))
        sgnflag = -1 if loc in dcands1 else 1
        direc = (sgnflag * (0 if lrflag else 1), sgnflag * (0 if not lrflag else 1))
        ln = shoot(loc, direc)
        shell = set()
        for k in range(min(oh, ow)-1):
            shell |= power(outbox, k+1)(ln)
        sqc, dotc = sample(ccols, 2)
        giobj = recolor(sqc, remove(loc, bd)) | {(dotc, loc)}
        goobj = recolor(sqc, (bd | shell) - ln) | recolor(dotc, ln)
        goobj = sfilter(goobj, lambda cij: cij[1] in fullinds)
        goobji = toindices(goobj)
        if goobji.issubset(inds):
            succ += 1
            inds = (inds - goobji) - mapply(dneighbors, bd)
            gi = paint(gi, giobj)
            go = paint(go, goobj)
    return {'input': gi, 'output': go}


def generate_228f6490(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    nsq = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 5 * nsq
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    sqc = choice(remcols)
    remcols = remove(sqc, remcols)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    forbidden = []
    while tr < maxtr and succ < nsq:
        tr += 1
        oh = randint(3, 6)
        ow = randint(3, 6)
        bd = asindices(canvas(-1, (oh, ow)))
        bounds = shift(asindices(canvas(-1, (oh-2, ow-2))), (1, 1))
        obj = {choice(totuple(bounds))}
        ncells = randint(1, (oh-2) * (ow-2))
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        sqcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(sqcands) == 0:
            continue
        loc = choice(totuple(sqcands))
        bdplcd = shift(bd, loc)
        if bdplcd.issubset(inds):
            tmpinds = inds - bdplcd
            inobjn = normalize(obj)
            oh, ow = shape(obj)
            inobjcands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(inobjcands) == 0:
                continue
            loc2 = choice(totuple(inobjcands))
            inobjplcd = shift(inobjn, loc2)
            bdnorm = bd - obj
            if inobjplcd.issubset(tmpinds) and bdnorm not in forbidden and inobjn not in forbidden:
                forbidden.append(bdnorm)
                forbidden.append(inobjn)
                succ += 1
                inds = (inds - (bdplcd | inobjplcd)) - mapply(dneighbors, inobjplcd)
                col = choice(remcols)
                oplcd = shift(obj, loc)
                gi = fill(gi, sqc, bdplcd - oplcd)
                go = fill(go, sqc, bdplcd)
                go = fill(go, col, oplcd)
                gi = fill(gi, col, inobjplcd)
    nremobjs = unifint(diff_lb, diff_ub, (0, len(inds) // 25))
    succ = 0
    tr = 0
    maxtr = 10 * nremobjs
    while tr < maxtr and succ < nremobjs:
        tr += 1
        oh = randint(1, 4)
        ow = randint(1, 4)
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncells = randint(1, oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        if obj in forbidden:
            continue
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        if plcd.issubset(inds):
            succ += 1
            inds = (inds - plcd) - mapply(dneighbors, plcd)
            col = choice(remcols)
            gi = fill(gi, col, plcd)
            go = fill(go, col, plcd)
    return {'input': gi, 'output': go}


def generate_93b581b8(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    numcols = unifint(diff_lb, diff_ub, (1, 9))
    ccols = sample(remcols, numcols)
    numocc = unifint(diff_lb, diff_ub, (1, (h * w) // 50))
    succ = 0
    tr = 0
    maxtr = 10 * numocc
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    inds = asindices(gi)
    fullinds = asindices(gi)
    while tr < maxtr and succ < numocc:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        c1, c2, c3, c4 = [choice(ccols) for k in range(4)]
        q = {(0, 0), (0, 1), (1, 0), (1, 1)}
        inobj = {(c1, (0, 0)), (c2, (0, 1)), (c3, (1, 0)), (c4, (1, 1))}
        outobj = inobj | recolor(c4, shift(q, (-2, -2))) | recolor(c3, shift(q, (-2, 2))) | recolor(c2, shift(q, (2, -2))) | recolor(c1, shift(q, (2, 2)))
        inobjplcd = shift(inobj, loc)
        outobjplcd = shift(outobj, loc)
        outobjplcd = sfilter(outobjplcd, lambda cij: cij[1] in fullinds)
        outobjplcdi = toindices(outobjplcd)
        if outobjplcdi.issubset(inds):
            succ += 1
            inds = (inds - outobjplcdi) - mapply(dneighbors, toindices(inobjplcd))
            gi = paint(gi, inobjplcd)
            go = paint(go, outobjplcd)
    return {'input': gi, 'output': go}


def generate_447fd412(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (12, 30))
    w = unifint(diff_lb, diff_ub, (12, 30))
    bgc, indic, mainc = sample(cols, 3)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    if oh * ow < 3:
        if choice((True, False)):
            oh = unifint(diff_lb, diff_ub, (3, 4))
        else:
            ow = unifint(diff_lb, diff_ub, (3, 4))
    bounds = asindices(canvas(-1, (oh, ow)))
    ncells = unifint(diff_lb, diff_ub, (3, oh * ow))
    obj = {choice(totuple(bounds))}
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objt = totuple(obj)
    kk = len(obj)
    nindic = randint(1, kk // 2 if kk % 2 == 1 else kk // 2 - 1)
    indicobj = set(sample(objt, nindic))
    mainobj = obj - indicobj
    obj = recolor(indic, indicobj) | recolor(mainc, mainobj)
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (h, w))
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = paint(go, plcd)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    fullinds = asindices(gi)
    noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (4 * len(plcd)))))
    tr = 0
    maxtr = 5 * noccs
    succ = 0
    while succ < noccs and tr < maxtr:
        tr += 1
        fac = randint(1, min(5, min(h, w) // max(oh, ow) - 1))
        outobj = upscale(obj, fac)
        inobj = sfilter(outobj, lambda cij: cij[0] == indic)
        hh, ww = shape(outobj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        inobjp = shift(inobj, loc)
        outobjp = shift(outobj, loc)
        outobjp = sfilter(outobjp, lambda cij: cij[1] in fullinds)
        outobjpi = toindices(outobjp)
        if outobjpi.issubset(inds):
            succ += 1
            inds = (inds - outobjpi) - mapply(neighbors, toindices(inobjp))
            gi = paint(gi, inobjp)
            go = paint(go, outobjp)
    return {'input': gi, 'output': go}


def generate_50846271(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(8, interval(0, 10, 1))
    cf1 = lambda d: {(d//2, 0), (d//2, d-1)} | set(sample(totuple(connect((d//2, 0), (d//2, d-1))), randint(1, d)))
    cf2 = lambda d: {(0, d//2), (d - 1, d//2)} | set(sample(totuple(connect((0, d//2), (d-1, d//2))), randint(1, d)))
    cf3 = lambda d: set(sample(totuple(remove((d//2, d//2), connect((d//2, 0), (d//2, d-1)))), randint(1, d-1))) | set(sample(totuple(remove((d//2, d//2), connect((0, d//2), (d - 1, d//2)))), randint(1, d-1)))
    cf = lambda d: choice((cf1, cf2, cf3))(d)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    dim = unifint(diff_lb, diff_ub, (1, 3))
    dim = 2 * dim + 1
    cross = connect((dim//2, 0), (dim//2, dim - 1)) | connect((0, dim//2), (dim - 1, dim//2))
    bgc, crossc, noisec = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    namt = unifint(diff_lb, diff_ub, (int(0.35 * h * w), int(0.65 * h * w)))
    inds = asindices(gi)
    noise = sample(totuple(inds), namt)
    gi = fill(gi, noisec, noise)
    initcross = choice((cf1, cf2))(dim)
    loci = randint(0, h - dim)
    locj = randint(0, w - dim)
    delt = shift(cross - initcross, (loci, locj))
    gi = fill(gi, crossc, shift(initcross, (loci, locj)))
    gi = fill(gi, noisec, delt)
    go = fill(gi, 8, delt)
    plcd = shift(cross, (loci, locj))
    bd = backdrop(plcd)
    nbhs = mapply(neighbors, plcd)
    inds = (inds - plcd) - nbhs
    nbhs2 = mapply(neighbors, nbhs)
    inds = inds - nbhs2
    inds = inds - mapply(neighbors, nbhs2)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) / (10 * dim)))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    while succ < noccs and tr < maxtr:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - dim and ij[1] <= w - dim)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        marked = shift(cf(dim), loc)
        full = shift(cross, loc)
        unmarked = full - marked
        inobj = recolor(noisec, unmarked) | recolor(crossc, marked)
        outobj = recolor(8, unmarked) | recolor(crossc, marked)
        outobji = toindices(outobj)
        if outobji.issubset(inds):
            dnbhs = mapply(neighbors, outobji)
            dnbhs2 = mapply(neighbors, dnbhs)
            inds = (inds - outobji) - (dnbhs | dnbhs2 | mapply(neighbors, dnbhs2))
            succ += 1
            gi = paint(gi, inobj)
            go = paint(go, outobj)
    return {'input': gi, 'output': go}


def generate_ae3edfdc(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2, 3, 7))
    h = unifint(diff_lb, diff_ub, (8, 30))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc = choice(cols)
    go = canvas(bgc, (h, w))
    inds = asindices(go)
    rdi = randint(1, h - 2)
    rdj = randint(1, w - 2)
    rd = (rdi, rdj)
    reminds = inds - ({rd} | neighbors(rd))
    reminds = sfilter(reminds, lambda ij: 1 <= ij[0] <= h - 2 and 1 <= ij[1] <= w - 2)
    bd = choice(totuple(reminds))
    bdi, bdj = bd
    go = fill(go, 2, {rd})
    go = fill(go, 1, {bd})
    ngd = unifint(diff_lb, diff_ub, (1, 8))
    gd = sample(totuple(neighbors(rd)), ngd)
    nod = unifint(diff_lb, diff_ub, (1, 8))
    od = sample(totuple(neighbors(bd)), nod)
    go = fill(go, 3, gd)
    go = fill(go, 7, od)
    gdmapper = {d: (3, position({rd}, {d})) for d in gd}
    odmapper = {d: (7, position({bd}, {d})) for d in od}
    mpr = {**gdmapper, **odmapper}
    ub = (len(gd) + len(od)) * ((h + w) // 5)
    ndist = unifint(diff_lb, diff_ub, (1, ub))
    gi = tuple(e for e in go)
    fullinds = asindices(gi)
    for k in range(ndist):
        options = []
        for loc, (col, direc) in mpr.items():
            ii, jj = add(loc, direc)
            if (ii, jj) in fullinds and gi[ii][jj] == bgc:
                options.append((loc, col, direc))
        if len(options) == 0:
            break
        loc, col, direc = choice(options)
        del mpr[loc]
        newloc = add(loc, direc)
        mpr[newloc] = (col, direc)
        gi = fill(gi, bgc, {loc})
        gi = fill(gi, col, {newloc})
    return {'input': gi, 'output': go}


def generate_469497ad(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(2, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (3, 6))
    w = unifint(diff_lb, diff_ub, (3, 6))
    bgc, sqc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    sqh = randint(1, h - 2)
    sqw = randint(1, w - 2)
    sqloci = randint(0, h - sqh - 2)
    sqlocj = randint(0, w - sqw - 2)
    sq = backdrop(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, sqc, sq)
    numcub = min(min(min(h, w)+1, 30//(max(h, w))), 7)
    numc = unifint(diff_lb, diff_ub, (2, numcub))
    numaccc = numc - 1
    remcols = remove(bgc, remove(sqc, cols))
    ccols = sample(remcols, numaccc)
    gi = rot180(gi)
    locs = sample(interval(1, min(h, w), 1), numaccc - 1)
    locs = [0] + sorted(locs)
    for c, l in zip(ccols, locs):
        gi = fill(gi, c, shoot((0, l), (0, 1)))
        gi = fill(gi, c, shoot((l, 0), (1, 0)))
    gi = rot180(gi)
    go = upscale(gi, numc)
    rect = ofcolor(go, sqc)
    l1 = shoot(lrcorner(rect), (1, 1))
    l2 = shoot(ulcorner(rect), (-1, -1))
    l3 = shoot(urcorner(rect), (-1, 1))
    l4 = shoot(llcorner(rect), (1, -1))
    ll = l1 | l2 | l3 | l4
    go = fill(go, 2, ll & ofcolor(go, bgc))
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_97a05b5b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (15, 30))
    w = unifint(diff_lb, diff_ub, (15, 30))
    sgh = randint(h//3, h//3*2)
    sgw = randint(w//3, w//3*2)
    bgc, sqc = sample(cols, 2)
    remcols = remove(bgc, remove(sqc, cols))
    gi = canvas(bgc, (h, w))
    oh = randint(2, sgh//2)
    ow = randint(2, sgw//2)
    nobjs = unifint(diff_lb, diff_ub, (1, 8))
    objs = set()
    cands = asindices(canvas(-1, (oh, ow)))
    forbidden = set()
    tr = 0
    maxtr = 4 * nobjs
    while len(objs) != nobjs and tr < maxtr:
        tr += 1
        obj = {choice(totuple(cands))}
        ncells = randint(1, oh * ow - 1)
        for k in range(ncells - 1):
            obj.add(choice(totuple((cands - obj) & mapply(neighbors, obj))))
        obj |= choice((dmirror, cmirror, vmirror, hmirror))(obj)
        if len(obj) == height(obj) * width(obj):
            continue
        obj = frozenset(obj)
        objn = normalize(obj)
        if objn not in forbidden:
            objs.add(objn)
        for augmf1 in (identity, dmirror, cmirror, hmirror, vmirror):
            for augmf2 in (identity, dmirror, cmirror, hmirror, vmirror):
                forbidden.add(augmf1(augmf2(objn)))
    tr = 0
    maxtr = 5 * nobjs
    succ = 0
    loci = randint(0, h - sgh)
    locj = randint(0, w - sgw)
    bd = backdrop(frozenset({(loci, locj), (loci + sgh - 1, locj + sgw - 1)}))
    gi = fill(gi, sqc, bd)
    go = canvas(sqc, (sgh, sgw))
    goinds = asindices(go)
    giinds = asindices(gi) - shift(goinds, (loci, locj))
    giinds = giinds - mapply(neighbors, shift(goinds, (loci, locj)))
    while succ < nobjs and tr < maxtr and len(objs) > 0:
        tr += 1
        obj = choice(totuple(objs))
        col = choice(remcols)
        subgi = fill(canvas(col, shape(obj)), sqc, obj)
        if len(palette(subgi)) == 1:
            continue
        f1 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f2 = choice((identity, dmirror, vmirror, cmirror, hmirror))
        f = compose(f1, f2)
        subgo = f(subgi)
        giobj = asobject(subgi)
        goobj = asobject(subgo)
        ohi, owi = shape(giobj)
        oho, owo = shape(goobj)
        gocands = sfilter(goinds, lambda ij: ij[0] <= sgh - oho and ij[1] <= sgw - owo)
        if len(gocands) == 0:
            continue
        goloc = choice(totuple(gocands))
        goplcd = shift(goobj, goloc)
        goplcdi = toindices(goplcd)
        if goplcdi.issubset(goinds):
            gicands = sfilter(giinds, lambda ij: ij[0] <= h - ohi and ij[1] <= owi)
            if len(gicands) == 0:
                continue
            giloc = choice(totuple(gicands))
            giplcd = shift(giobj, giloc)
            giplcdi = toindices(giplcd)
            if giplcdi.issubset(giinds):
                succ += 1
                remcols = remove(col, remcols)
                objs = remove(obj, objs)
                goinds = goinds - goplcdi
                giinds = (giinds - giplcdi) - mapply(neighbors, giplcdi)
                gi = paint(gi, giplcd)
                gi = fill(gi, bgc, sfilter(shift(goplcd, (loci, locj)), lambda cij: cij[0] == sqc))
                go = paint(go, goplcd)
    return {'input': gi, 'output': go}


def generate_a5313dff(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, fgc = sample(cols, 2)
    gi = canvas(bgc, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    inds = shift(asindices(canvas(-1, (h+2, w+2))), (-1, -1))
    while (tr < maxtr and succ < noccs) or len(sfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))) == 0:
        tr += 1
        oh = randint(3, 8)
        ow = randint(3, 8)
        bx = box(frozenset({(0, 0), (oh - 1, ow - 1)}))
        ins = backdrop(inbox(bx))
        loc = choice(totuple(inds))
        plcdins = shift(ins, loc)
        if len(plcdins & ofcolor(gi, fgc)) == 0:
            succ += 1
            gi = fill(gi, fgc, shift(bx, loc))
            if choice((True, True, False)):
                ss = sample(totuple(plcdins), randint(1, max(1, len(ins) // 2)))
                gi = fill(gi, fgc, ss)
    res = mfilter(colorfilter(objects(gi, T, F, F), bgc), compose(flip, rbind(bordering, gi)))
    go = fill(gi, 1, res)
    return {'input': gi, 'output': go}


def generate_780d0b14(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    nh = unifint(diff_lb, diff_ub, (2, 6))
    nw = unifint(diff_lb, diff_ub, (2, 6))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    ncols = unifint(diff_lb, diff_ub, (3, 9))
    ccols = sample(remcols, ncols)
    go = canvas(-1, (nh, nw))
    obj = {(choice(ccols), ij) for ij in asindices(go)}
    go = paint(go, obj)
    while len(dedupe(go)) < nh or len(dedupe(dmirror(go))) < nw:
        obj = {(choice(ccols), ij) for ij in asindices(go)}
        go = paint(go, obj)
    h = unifint(diff_lb, diff_ub, (2*nh+nh-1, 30))
    w = unifint(diff_lb, diff_ub, (2*nw+nw-1, 30))
    hdist = [2 for k in range(nh)]
    for k in range(h - 2 * nh - nh + 1):
        idx = randint(0, nh - 1)
        hdist[idx] += 1
    wdist = [2 for k in range(nw)]
    for k in range(w - 2 * nw - nw + 1):
        idx = randint(0, nw - 1)
        wdist[idx] += 1
    gi = merge(tuple(repeat(r, c) + (repeat(bgc, nw),) for r, c in zip(go, hdist)))[:-1]
    gi = dmirror(merge(tuple(repeat(r, c) + (repeat(bgc, h),) for r, c in zip(dmirror(gi), wdist)))[:-1])
    objs = objects(gi, T, F, F)
    bgobjs = colorfilter(objs, bgc)
    objs = objs - bgobjs
    for obj in objs:
        gi = fill(gi, bgc, sample(totuple(toindices(obj)), unifint(diff_lb, diff_ub, (1, len(obj) // 2))))
    return {'input': gi, 'output': go}


def generate_57aa92db(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(2, 5)
    ow = randint(2, 5)
    bounds = asindices(canvas(-1, (oh, ow)))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fixp = choice(totuple(obj))
    bgc, fixc, mainc = sample(cols, 3)
    remcols = difference(cols, (bgc, fixc, mainc))
    gi = canvas(bgc, (h, w))
    obj = {(fixc, fixp)} | recolor(mainc, remove(fixp, obj))
    loci = randint(0, h - oh)
    locj = randint(0, w - ow)
    plcd = shift(obj, (loci, locj))
    gi = paint(gi, plcd)
    go = tuple(e for e in gi)
    inds = ofcolor(gi, bgc) - mapply(neighbors, toindices(plcd))
    nocc = unifint(diff_lb, diff_ub, (1, (h * w) // (4 * len(obj))))
    tr = 0
    succ = 0
    maxtr = 5 * nocc
    while succ < nocc and tr < maxtr:
        tr += 1
        fac = randint(1, 4)
        objups = upscale(obj, fac)
        hh, ww = shape(objups)
        cands = sfilter(inds, lambda ij: ij[0] <= h - hh and ij[1] <= w - ww)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        objupsplcd = shift(objups, loc)
        objupsplcdi = toindices(objupsplcd)
        if objupsplcdi.issubset(inds):
            succ += 1
            newc = choice(remcols)
            fixp2 = sfilter(objupsplcd, lambda cij: cij[0] == fixc)
            inds = inds - mapply(neighbors, objupsplcdi)
            gi = paint(gi, fixp2)
            go = paint(go, fixp2)
            remobjfull = toindices(objupsplcd - fixp2)
            ntorem = unifint(diff_lb, diff_ub, (0, max(0, len(remobjfull) - 1)))
            ntokeep = len(remobjfull) - ntorem
            tokeep = {choice(totuple(remobjfull & outbox(fixp2)))}
            fixp2i = toindices(fixp2)
            for k in range(ntokeep - 1):
                fullopts = remobjfull & mapply(neighbors, tokeep | fixp2i)
                remopts = fullopts - tokeep
                tokeep.add(choice(totuple(remopts)))
            gi = fill(gi, newc, tokeep)
            go = fill(go, newc, remobjfull)
    return {'input': gi, 'output': go}


def generate_53b68214(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (2, 6))
        w = unifint(diff_lb, diff_ub, (8, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        ncols = unifint(diff_lb, diff_ub, (1, 9))
        ccols = sample(remcols, ncols)
        oh = unifint(diff_lb, diff_ub, (1, h//2))
        ow = unifint(diff_lb, diff_ub, (1, w//2-1))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (1, oh * ow))
        obj = sample(totuple(bounds), ncells)
        obj = {(choice(ccols), ij) for ij in obj}
        obj = normalize(obj)
        oh, ow = shape(obj)
        locj = randint(0, w//2)
        plcd = shift(obj, (0, locj))
        go = canvas(bgc, (10, w))
        hoffs = randint(0, ow//2 + 1)
        for k in range(10//oh+1):
            go = paint(go, shift(plcd, (k*oh, k*hoffs)))
        if len(palette(go[h:])) > 1:
            break
    gi = go[:h]
    if choice((True, False)):
        gi = vmirror(gi)
        go = vmirror(go)
    return {'input': gi, 'output': go}


def generate_39e1d7f9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (5, 10))
    w = unifint(diff_lb, diff_ub, (5, 10))
    bgc, linc, dotc = sample(cols, 3)
    remcols = difference(cols, (bgc, linc, dotc))
    gi = canvas(bgc, (h, w))
    loci = randint(1, h - 2)
    locj = randint(1, w - 2)
    if h == 5:
        loci = choice((1, h - 2))
    if w == 5:
        locj = choice((1, w - 2))
    npix = unifint(diff_lb, diff_ub, (1, 8))
    ncols = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, ncols)
    candsss = neighbors((loci, locj))
    pixs = {(loci, locj)}
    for k in range(npix):
        pixs.add(choice(totuple((mapply(dneighbors, pixs) & candsss) - pixs)))
    pixs = totuple(remove((loci, locj), pixs))
    obj = {(choice(ccols), ij) for ij in pixs}
    gi = fill(gi, dotc, {(loci, locj)})
    gi = paint(gi, obj)
    go = tuple(e for e in gi)
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // (2 * len(pixs) + 1)))
    succ = 0
    tr = 0
    maxtr = 6 * noccs
    inds = ofcolor(gi, bgc) - mapply(dneighbors, neighbors((loci, locj)))
    objn = shift(obj, (-loci, -locj))
    triedandfailed = set()
    while (tr < maxtr and succ < noccs) or succ == 0:
        lopcands = totuple(inds - triedandfailed)
        if len(lopcands) == 0:
            break
        tr += 1
        loci, locj = choice(lopcands)
        plcd = shift(objn, (loci, locj))
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            inds = inds - (plcdi | {(loci, locj)})
            succ += 1
            gi = fill(gi, dotc, {(loci, locj)})
            go = fill(go, dotc, {(loci, locj)})
            go = paint(go, plcd)
        else:
            triedandfailed.add((loci, locj))
    hfac = unifint(diff_lb, diff_ub, (1, (30 - h + 1) // h))
    wfac = unifint(diff_lb, diff_ub, (1, (30 - w + 1) // w))
    fullh = hfac * h + h - 1
    fullw = wfac * w + w - 1
    gi2 = canvas(linc, (fullh, fullw))
    go2 = canvas(linc, (fullh, fullw))
    bd = asindices(canvas(-1, (hfac, wfac)))
    for a in range(h):
        for b in range(w):
            c = gi[a][b]
            gi2 = fill(gi2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    for a in range(h):
        for b in range(w):
            c = go[a][b]
            go2 = fill(go2, c, shift(bd, (a * (hfac + 1), b * (wfac + 1))))
    gi, go = gi2, go2
    return {'input': gi, 'output': go}


def generate_017c7c7b(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (0, 2))
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = unifint(diff_lb, diff_ub, (2, 30))
    h += h
    fgc = choice(cols)
    go = canvas(0, (h + h // 2, w))
    oh = unifint(diff_lb, diff_ub, (1, h//3*2))
    ow = unifint(diff_lb, diff_ub, (1, w))
    locj = randint(0, w - ow)
    bounds = asindices(canvas(-1, (oh, ow)))
    ncellsd = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(1, ncells), oh * ow)
    obj = sample(totuple(bounds), ncells)
    for k in range((2*h)//oh):
        go = fill(go, 2, shift(obj, (k*oh, 0)))
    gi = replace(go[:h], 2, fgc)
    return {'input': gi, 'output': go}


def generate_8a004b2b(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = unifint(diff_lb, diff_ub, (2, h//5))
    ow = unifint(diff_lb, diff_ub, (2, w//5))
    bounds = asindices(canvas(-1, (oh, ow)))
    bgc, cornc, ac1, ac2, objc = sample(cols, 5)
    gi = canvas(bgc, (h, w))
    obj = {choice(totuple(bounds))}
    ncellsd = unifint(diff_lb, diff_ub, (0, (oh * ow) // 2))
    ncells = choice((ncellsd, oh * ow - ncellsd))
    ncells = min(max(3, ncells), oh * ow)
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    fp1 = choice(totuple(obj))
    fp2 = choice(remove(fp1, totuple(obj)))
    remobj = obj - {fp1, fp2}
    obj = recolor(objc, remobj) | {(ac1, fp1), (ac2, fp2)}
    maxhscf = (h - oh - 4) // oh
    maxwscf = (w - ow - 4) // ow
    hscf = unifint(diff_lb, diff_ub, (1, maxhscf))
    wscf = unifint(diff_lb, diff_ub, (1, maxwscf))
    loci = randint(0, 2)
    locj = randint(0, 2)
    oplcd = shift(obj, (loci, locj))
    gi = paint(gi, oplcd)
    inh = hscf * oh
    inw = wscf * ow
    sqh = unifint(diff_lb, diff_ub, (inh + 2, h - oh - 2))
    sqw = unifint(diff_lb, diff_ub, (inw + 2, w))
    sqloci = randint(loci+oh, h - sqh)
    sqlocj = randint(0, w - sqw)
    crns = corners(frozenset({(sqloci, sqlocj), (sqloci + sqh - 1, sqlocj + sqw - 1)}))
    gi = fill(gi, cornc, crns)
    gomini = subgrid(oplcd, gi)
    goo = vupscale(hupscale(gomini, wscf), hscf)
    goo = asobject(goo)
    gloci = randint(sqloci+1, sqloci+sqh-1-height(goo))
    glocj = randint(sqlocj+1, sqlocj+sqw-1-width(goo))
    gooplcd = shift(goo, (gloci, glocj))
    go = paint(gi, gooplcd)
    go = subgrid(crns, go)
    indic = sfilter(gooplcd, lambda cij: cij[0] in (ac1, ac2))
    gi = paint(gi, indic)
    if choice((True, False)) and len(obj) > 3:
        idx = choice(totuple(toindices(sfilter(obj, lambda cij: cij[0] == objc))))
        idxi, idxj = idx
        xx = shift(asindices(canvas(-1, (hscf, wscf))), (gloci+idxi*hscf, glocj+idxj*wscf))
        gi = fill(gi, objc, xx)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_49d1d64f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (2, 28))
    w = unifint(diff_lb, diff_ub, (2, 28))
    ncols = unifint(diff_lb, diff_ub, (1, 10))
    ccols = sample(cols, ncols)
    gi = canvas(-1, (h, w))
    obj = {(choice(ccols), ij) for ij in asindices(gi)}
    gi = paint(gi, obj)
    go = canvas(0, (h+2, w+2))
    go = paint(go, shift(asobject(gi), (1, 1)))
    ts = sfilter(obj, lambda cij: cij[1][0] == 0)
    bs = sfilter(obj, lambda cij: cij[1][0] == h - 1)
    ls = sfilter(obj, lambda cij: cij[1][1] == 0)
    rs = sfilter(obj, lambda cij: cij[1][1] == w - 1)
    ts = shift(ts, (1, 1))
    bs = shift(bs, (1, 1))
    ls = shift(ls, (1, 1))
    rs = shift(rs, (1, 1))
    go = paint(go, shift(ts, (-1, 0)))
    go = paint(go, shift(bs, (1, 0)))
    go = paint(go, shift(ls, (0, -1)))
    go = paint(go, shift(rs, (0, 1)))
    return {'input': gi, 'output': go}


def generate_890034e9(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(1, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    oh = randint(2, h//4)
    ow = randint(2, w//4)
    markercol = choice(cols)
    remcols = remove(markercol, cols)
    numbgc = unifint(diff_lb, diff_ub, (1, 8))
    bgcols = sample(remcols, numbgc)
    gi = canvas(0, (h, w))
    inds = asindices(gi)
    obj = {(choice(bgcols), ij) for ij in inds}
    gi = paint(gi, obj)
    numbl = unifint(diff_lb, diff_ub, (0, (h * w) // 2))
    blacks = sample(totuple(inds), numbl)
    gi = fill(gi, 0, blacks)
    patt = asindices(canvas(-1, (oh, ow)))
    tocover = set()
    for occ in occurrences(gi, recolor(0, patt)):
        tocover.add(choice(totuple(shift(patt, occ))))
    tocover = {(choice(bgcols), ij) for ij in tocover}
    gi = paint(gi, tocover)
    noccs = unifint(diff_lb, diff_ub, (2, (h * w) // ((oh + 2) * (ow + 2))))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        bd = shift(patt, loc)
        plcd = outbox(bd)
        if plcd.issubset(inds):
            succ += 1
            inds = inds - plcd
            gi = fill(gi, 0, bd)
            go = fill(go, 0, bd)
            if succ == 1:
                gi = fill(gi, markercol, plcd)
            go = fill(go, markercol, plcd)
            loci, locj = loc
            ln1 = connect((loci-1, locj), (loci-1, locj+ow-1))
            ln2 = connect((loci+oh, locj), (loci+oh, locj+ow-1))
            ln3 = connect((loci, locj-1), (loci+oh-1, locj-1))
            ln4 = connect((loci, locj+ow), (loci+oh-1, locj+ow))
            if succ > 1:
                fixxer = {
                    (choice(bgcols), choice(totuple(xx))) for xx in [ln1, ln2, ln3, ln4]
                }
                gi = paint(gi, fixxer)
    return {'input': gi, 'output': go}


def generate_776ffc46(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc, sqc, inc, outc = sample(cols, 4)
    gi = canvas(bgc, (h, w))
    sqh = randint(3, h//3+1)
    sqw = randint(3, w//3+1)
    loci = randint(0, 3)
    locj = randint(0, w - sqw)
    bx = box(frozenset({(loci, locj), (loci + sqh - 1, locj + sqw - 1)}))
    bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
    obj = {choice(totuple(bounds))}
    ncells = randint(1, (sqh - 2) * (sqw - 2))
    for k in range(ncells - 1):
        obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
    obj = normalize(obj)
    oh, ow = shape(obj)
    objp = shift(obj, (loci+1+randint(0, sqh-oh-2), locj+1+randint(0, sqw-ow-2)))
    gi = fill(gi, sqc, bx)
    gi = fill(gi, inc, objp)
    inds = (ofcolor(gi, bgc) - backdrop(bx)) - mapply(neighbors, backdrop(bx))
    cands = sfilter(inds, lambda ij: shift(obj, ij).issubset(inds))
    loc = choice(totuple(cands))
    plcd = shift(obj, loc)
    gi = fill(gi, outc, plcd)
    inds = (inds - plcd) - mapply(neighbors, plcd)
    noccs = unifint(diff_lb, diff_ub, (0, (h * w) // 20))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    fullinds = asindices(gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        if choice((True, False)):
            sqh = randint(3, h//3+1)
            sqw = randint(3, w//3+1)
            bx = box(frozenset({(0, 0), (sqh - 1, sqw - 1)}))
            bounds = asindices(canvas(-1, (sqh - 2, sqw - 2)))
            obj2 = {choice(totuple(bounds))}
            ncells = randint(1, (sqh - 2) * (sqw - 2))
            for k in range(ncells - 1):
                obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < (sqh - 2) * (sqw - 2):
                    obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
            obj2 = normalize(obj2)
            ooh, oow = shape(obj2)
            cands1 = connect((-1, -1), (-1, w - sqw + 1))
            cands2 = connect((h-sqh+1, -1), (h-sqh+1, w - sqw + 1))
            cands3 = connect((-1, -1), (h - sqh + 1, -1))
            cands4 = connect((-1, w-sqw+1), (h - sqh + 1, w-sqw+1))
            cands = cands1 | cands2 | cands3 | cands4
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            sloci, slocj = loc
            plcdbx = shift(bx, loc)
            if (backdrop(plcdbx) & fullinds).issubset(inds):
                succ += 1
                oloci = randint(sloci+1, sloci+1+randint(0, sqh-ooh-2))
                olocj = randint(slocj+1, slocj+1+randint(0, sqw-oow-2))
                gi = fill(gi, sqc, plcdbx)
                gi = fill(gi, inc, shift(obj2, (oloci, olocj)))
                inds = inds - backdrop(outbox(plcdbx))
        else:
            ooh = randint(1, h//3-1)
            oow = randint(1, w//3-1)
            bounds = asindices(canvas(-1, (ooh, oow)))
            obj2 = {choice(totuple(bounds))}
            ncells = randint(1, oow * ooh)
            for k in range(ncells - 1):
                obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
            if normalize(obj2) == obj:
                if len(obj2) < ooh * oow:
                    obj2.add(choice(totuple((bounds - obj2) & mapply(dneighbors, obj2))))
                else:
                    continue
        if choice((True, False, False)):
            obj2 = obj
        obj2 = normalize(obj2)
        ooh, oow = shape(obj2)
        for kk in range(randint(1, 3)):
            cands = sfilter(inds, lambda ij: ij[0] <= h - ooh and ij[1] <= w - oow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            plcd = shift(obj2, loc)
            if plcd.issubset(inds):
                succ += 1
                inds = (inds - plcd) - mapply(neighbors, plcd)
                gi = fill(gi, outc, plcd)
    objs = objects(gi, T, F, F)
    objs = colorfilter(objs, outc)
    objs = mfilter(objs, lambda o: equality(normalize(toindices(o)), obj))
    go = fill(gi, inc, objs)
    return {'input': gi, 'output': go}


def generate_e6721834(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 15))
    w = unifint(diff_lb, diff_ub, (8, 30))
    bgc1, bgc2, sqc = sample(cols, 3)
    remcols = difference(cols, (bgc1, bgc2, sqc))
    gi1 = canvas(bgc1, (h, w))
    gi2 = canvas(bgc2, (h, w))
    noccs = unifint(diff_lb, diff_ub, (1, (h * w) // 16))
    tr = 0
    succ = 0
    maxtr = 5 * noccs
    gi1inds = asindices(gi1)
    gi2inds = asindices(gi2)
    go = canvas(bgc2, (h, w))
    seen = []
    while tr < maxtr and succ < noccs:
        tr += 1
        oh = randint(2, min(6, h//2))
        ow = randint(2, min(6, w//2))
        cands = sfilter(gi1inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        bounds = shift(asindices(canvas(-1, (oh, ow))), loc)
        ncells = unifint(diff_lb, diff_ub, (1, (oh * ow) // 2))
        obj = set(sample(totuple(bounds), ncells))
        objc = choice(remcols)
        objn = normalize(obj)
        if (objn, objc) in seen:
            continue
        seen.append(((objn, objc)))
        if bounds.issubset(gi1inds):
            succ += 1
            gi1inds = (gi1inds - bounds) - mapply(neighbors, bounds)
            gi1 = fill(gi1, sqc, bounds)
            gi1 = fill(gi1, objc, obj)
            cands2 = sfilter(gi2inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands2) == 0:
                continue
            loc2 = choice(totuple(cands2))
            bounds2 = shift(shift(bounds, invert(loc)), loc2)
            obj2 = shift(shift(obj, invert(loc)), loc2)
            if bounds2.issubset(gi2inds):
                gi2inds = (gi2inds - bounds2) - mapply(neighbors, bounds2)
                gi2 = fill(gi2, objc, obj2)
                go = fill(go, sqc, bounds2)
                go = fill(go, objc, obj2)
    gi = vconcat(gi1, gi2)
    mfs = (identity, dmirror, cmirror, vmirror, hmirror, rot90, rot180, rot270)
    nmfs = choice((1, 2))
    for fn in sample(mfs, nmfs):
        gi = fn(gi)
        go = fn(go)
    return {'input': gi, 'output': go}


def generate_ef135b50(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(9, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        numc = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, numc)
        gi = canvas(bgc, (h, w))
        nsq = unifint(diff_lb, diff_ub, (2, (h * w) // 30))
        succ = 0
        tr = 0
        maxtr = 5 * nsq
        inds = asindices(gi)
        pats = set()
        while tr < maxtr and succ < nsq:
            tr += 1
            oh = randint(1, (h//3*2))
            ow = randint(1, (w//3*2))
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            loci, locj = loc
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            if bd.issubset(inds):
                succ += 1
                inds = (inds - bd) - mapply(neighbors, bd)
                gi = fill(gi, choice(ccols), bd)
                pats.add(bd)
        res = set()
        ofc = ofcolor(gi, bgc)
        for pat1 in pats:
            for pat2 in remove(pat1, pats):
                if hmatching(pat1, pat2):
                    um = max(uppermost(pat1), uppermost(pat2))
                    bm = min(lowermost(pat1), lowermost(pat2))
                    lm = min(rightmost(pat1), rightmost(pat2)) + 1
                    rm = max(leftmost(pat1), leftmost(pat2)) - 1
                    res = res | backdrop(frozenset({(um, lm), (bm, rm)}))
        res = (res & ofc) - box(asindices(gi))
        go = fill(gi, 9, res)
        if go != gi:
            break
    return {'input': gi, 'output': go}


def generate_794b24be(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 1)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nblue = randint(1, 4)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 2, {mpr[k+1]})
    gi = canvas(bgc, (h, w))
    locs = sample(totuple(asindices(gi)), nblue)
    gi = fill(gi, 1, locs)
    remlocs = ofcolor(gi, bgc)
    namt = unifint(diff_lb, diff_ub, (0, len(remlocs) // 2 - 1))
    remcols = remove(bgc, cols)
    numc = unifint(diff_lb, diff_ub, (1, 7))
    ccols = sample(remcols, numc)
    noise = sample(totuple(remlocs), namt)
    noise = {(choice(ccols), ij) for ij in noise}
    gi = paint(gi, noise)
    return {'input': gi, 'output': go}


def generate_ff28f65a(diff_lb: float, diff_ub: float) -> dict:
    cols = difference(interval(0, 10, 1), (1, 2))
    mpr = {1: (0, 0), 2: (0, 2), 3: (1, 1), 4: (2, 0), 5: (2, 2)}
    h = unifint(diff_lb, diff_ub, (3, 30))
    w = unifint(diff_lb, diff_ub, (3, 30))
    bgc = choice(cols)
    nred = randint(1, 5)
    gi = canvas(bgc, (h, w))
    succ = 0
    tr = 0
    maxtr = 5 * nred
    inds = asindices(gi)
    while tr < maxtr and succ < nred:
        tr += 1
        oh = randint(1, h//2+1)
        ow = randint(1, w//2+1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci+oh-1, locj+ow-1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, 2, bd)
    nblue = succ
    namt = unifint(diff_lb, diff_ub, (0, nred * 2))
    succ = 0
    tr = 0
    maxtr = 5 * namt
    remcols = remove(bgc, cols)
    tr += 1
    while tr < maxtr and succ < namt:
        tr += 1
        oh = randint(1, h//2+1)
        ow = randint(1, w//2+1)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        loci, locj = loc
        bd = backdrop(frozenset({(loci, locj), (loci+oh-1, locj+ow-1)}))
        if bd.issubset(inds):
            succ += 1
            inds = (inds - bd) - mapply(dneighbors, bd)
            gi = fill(gi, choice(remcols), bd)
    go = canvas(bgc, (3, 3))
    for k in range(nblue):
        go = fill(go, 1, {mpr[k+1]})
    return {'input': gi, 'output': go}


def generate_73251a56(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        d = unifint(diff_lb, diff_ub, (10, 30))
        h, w = d, d
        noisec = choice(cols)
        remcols = remove(noisec, cols)
        nsl = unifint(diff_lb, diff_ub, (2, min(9, h//2)))
        slopes = [0] + sorted(sample(interval(1, h-1, 1), nsl - 1))
        ccols = sample(cols, nsl)
        gi = canvas(-1, (h, w))
        inds = asindices(gi)
        for col, hdelt in zip(ccols, slopes):
            slope = hdelt / w
            locs = sfilter(inds, lambda ij: slope * ij[1] <= ij[0])
            gi = fill(gi, col, locs)
        ln = connect((0, 0), (d - 1, d - 1))
        gi = fill(gi, ccols[-2], ln)
        obj = asobject(gi)
        obj = sfilter(obj, lambda cij: cij[1][1] >= cij[1][0])
        gi = paint(gi, dmirror(obj))
        cf1 = lambda g: ccols[-2] in palette(toobject(ln, g))
        cf2 = lambda g: len((ofcolor(g, noisec) & frozenset({ij[::-1] for ij in ofcolor(g, noisec)})) - ln) == 0
        ndist = unifint(diff_lb, diff_ub, (1, (h * w) // 15))
        tr = 0
        succ = 0
        maxtr = 10 * ndist
        go = tuple(e for e in gi)
        while tr < maxtr and succ < ndist:
            tr += 1
            oh = randint(1, 5)
            ow = randint(1, 5)
            loci = randint(1, h - oh - 1)
            locj = randint(1, w - ow - 1)
            bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
            gi2 = fill(gi, noisec, bd)
            if cf1(gi2) and cf2(gi2):
                succ += 1
                gi = gi2
        if gi != go:
            break
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_3631a71a(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (6, 15))
    w = h
    bgc, patchcol = sample(cols, 2)
    patchcol = choice(cols)
    bgc = choice(remove(patchcol, cols))
    remcols = difference(cols, (bgc, patchcol))
    c = canvas(bgc, (h, w))
    inds = sfilter(asindices(c), lambda ij: ij[0] >= ij[1])
    ncols = unifint(diff_lb, diff_ub, (1, 8))
    ccols = sample(remcols, ncols)
    ncells = unifint(diff_lb, diff_ub, (1, len(inds)))
    cells = set(sample(totuple(inds), ncells))
    obj = {(choice(ccols), ij) for ij in cells}
    c = paint(dmirror(paint(c, obj)), obj)
    c = hconcat(c, vmirror(c))
    c = vconcat(c, hmirror(c))
    cutoff = 2
    go = dmirror(dmirror(c[:-cutoff])[:-cutoff])
    gi = tuple(e for e in go)
    forbidden = asindices(canvas(-1, (cutoff, cutoff)))
    dmirrareaL = shift(asindices(canvas(-1, (h*2-2*cutoff, cutoff))), (cutoff, 0))
    dmirrareaT = shift(asindices(canvas(-1, (cutoff, 2*w-2*cutoff))), (0, cutoff))
    inds1 = sfilter(asindices(gi), lambda ij: cutoff <= ij[0] < h and cutoff <= ij[1] < w and ij[0] >= ij[1])
    inds2 = dmirror(inds1)
    inds3 = shift(hmirror(inds1), (h-cutoff, 0))
    inds4 = shift(hmirror(inds2), (h-cutoff, 0))
    inds5 = shift(vmirror(inds1), (0, w-cutoff))
    inds6 = shift(vmirror(inds2), (0, w-cutoff))
    inds7 = shift(hmirror(vmirror(inds1)), (h-cutoff, w-cutoff))
    inds8 = shift(hmirror(vmirror(inds2)), (h-cutoff, w-cutoff))
    f1 = identity
    f2 = dmirror
    f3 = lambda x: hmirror(shift(x, invert((h-cutoff, 0))))
    f4 = lambda x: dmirror(hmirror(shift(x, invert((h-cutoff, 0)))))
    f5 = lambda x: vmirror(shift(x, invert((0, w-cutoff))))
    f6 = lambda x: dmirror(vmirror(shift(x, invert((0, w-cutoff)))))
    f7 = lambda x: vmirror(hmirror(shift(x, invert((h-cutoff, w-cutoff)))))
    f8 = lambda x: dmirror(vmirror(hmirror(shift(x, invert((h-cutoff, w-cutoff))))))
    indsarr = [inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8]
    farr = [f1, f2, f3, f4, f5, f6, f7, f8]
    ndist = unifint(diff_lb, diff_ub, (1, int((2*h*2*w) ** 0.5)))
    succ = 0
    tr = 0
    maxtr = 10 * ndist
    fullh, fullw = shape(gi)
    while succ < ndist and tr < maxtr:
        tr += 1
        oh = randint(2, h//2+1)
        ow = randint(2, w//2+1)
        loci = randint(0, fullh - oh)
        locj = randint(0, fullw - ow)
        bd = backdrop(frozenset({(loci, locj), (loci + oh - 1, locj + ow - 1)}))
        isleft = set()
        gi2 = fill(gi, patchcol, bd)
        if patchcol in palette(toobject(forbidden, gi2)):
            continue
        oo1 = toindices(sfilter(toobject(dmirrareaL, gi2), lambda cij: cij[0] != patchcol))
        oo2 = toindices(sfilter(toobject(dmirrareaT, gi2), lambda cij: cij[0] != patchcol))
        oo2 = frozenset({(ij[1], ij[0]) for ij in oo2})
        if oo1 | oo2 != dmirrareaL:
            continue
        for ii, ff in zip(indsarr, farr):
            oo = toobject(ii, gi2)
            rem = toindices(sfilter(oo, lambda cij: cij[0] != patchcol))
            if len(rem) > 0:
                isleft = isleft | ff(rem)
        if isleft != inds1:
            continue
        succ += 1
        gi = gi2
    return {'input': gi, 'output': go}


def generate_234bbc79(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (5, 30))
        w = unifint(diff_lb, diff_ub, (6, 20))
        bgc, dotc = sample(cols, 2)
        remcols = difference(cols, (bgc, dotc))
        go = canvas(bgc, (h, 30))
        ncols = unifint(diff_lb, diff_ub, (1, 8))
        ccols = sample(remcols, ncols)
        spi = randint(0, h - 1)
        snek = [(spi, 0)]
        gi = fill(go, dotc, {(spi, 0)})
        while True:
            previ, prevj = snek[-1]
            if prevj == w - 1:
                if choice((True, False, False)):
                    break
            options = []
            if previ < h - 1:
                if go[previ+1][prevj] == bgc:
                    options.append((previ+1, prevj))
            if previ > 0:
                if go[previ-1][prevj] == bgc:
                    options.append((previ-1, prevj))
            if prevj < w - 1:
                options.append((previ, prevj+1))
            if len(options) == 0:
                break
            loc = choice(options)
            snek.append(loc)
            go = fill(go, dotc, {loc})
        objs = []
        cobj = []
        for idx, cel in enumerate(snek):
            if len(cobj) > 2 and width(frozenset(cobj)) > 1 and snek[idx-1] == add(cel, (0, -1)):
                objs.append(cobj)
                cobj = [cel]
            else:
                cobj.append(cel)
        objs[-1] += cobj
        nobjs = len(objs)
        if nobjs < 2:
            continue
        ntokeep = unifint(diff_lb, diff_ub, (2, nobjs))
        ntorem = nobjs - ntokeep
        for k in range(ntorem):
            idx = randint(0, len(objs) - 2)
            objs = objs[:idx] + [objs[idx] + objs[idx+1]] + objs[idx+2:]
        inobjs = []
        for idx, obj in enumerate(objs):
            col = choice(ccols)
            go = fill(go, col, set(obj))
            centerpart = recolor(col, set(obj[1:-1]))
            leftpart = {(dotc if idx > 0 else col, obj[0])}
            rightpart = {(dotc if idx < len(objs) - 1 else col, obj[-1])}
            inobj = centerpart | leftpart | rightpart
            inobjs.append(inobj)
        spacings = [1 for idx in range(len(inobjs) - 1)]
        fullw = unifint(diff_lb, diff_ub, (w, 30))
        for k in range(fullw - w - len(inobjs) - 1):
            idx = randint(0, len(spacings) - 1)
            spacings[idx] += 1
        lspacings = [0] + spacings
        gi = canvas(bgc, (h, fullw))
        ofs = 0
        for i, (lsp, obj) in enumerate(zip(lspacings, inobjs)):
            obj = set(obj)
            if i == 0:
                ulc = ulcorner(obj)
            else:
                ulci = randint(0, h - height(obj))
                ulcj = ofs + lsp
                ulc = (ulci, ulcj)
            ofs += width(obj) + lsp
            plcd = shift(normalize(obj), ulc)
            gi = paint(gi, plcd)
        break
    ins = size(merge(fgpartition(gi)))
    while True:
        go2 = dmirror(dmirror(go)[:-1])
        if size(sfilter(asobject(go2), lambda cij: cij[0] != bgc)) < ins:
            break
        else:
            go = go2
    return {'input': gi, 'output': go}


def generate_cbded52d(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    for col in ccols:
        inds = ofcolor(go, bgc)
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        narms = randint(1, 4)
        armdirs = sample(totuple(dneighbors((0, 0))), narms)
        succ = 0
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, {endp})
            go = fill(go, col, set(arm[:aidx+1]))
            succ += 1
        if succ > 0:
            gi = fill(gi, col, {loc})
            go = fill(go, col, {loc})
    return {'input': gi, 'output': go}


def generate_06df4c85(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    oh = unifint(diff_lb, diff_ub, (1, 4))
    ow = unifint(diff_lb, diff_ub, (1, 4))
    numh = unifint(diff_lb, diff_ub, (3, 31 // (oh + 1)))
    numw = unifint(diff_lb, diff_ub, (3, 31 // (ow + 1)))
    bgc, linc = sample(cols, 2)
    remcols = difference(cols, (bgc, linc))
    ncols = unifint(diff_lb, diff_ub, (1, min(8, (numh * numh) // 3)))
    ccols = sample(remcols, ncols)
    fullh = numh * oh + numh - 1
    fullw = numw * ow + numw - 1
    gi = canvas(linc, (fullh, fullw))
    sgi = asindices(canvas(bgc, (oh, ow)))
    for a in range(numh):
        for b in range(numw):
            gi = fill(gi, bgc, shift(sgi, (a * (oh + 1), b * (ow + 1))))
    go = tuple(e for e in gi)
    sinds = asindices(canvas(-1, (oh, ow)))
    for col in ccols:
        inds = occurrences(go, recolor(bgc, sinds))
        if len(inds) == 0:
            break
        loc = choice(totuple(inds))
        narms = randint(1, 4)
        armdirs = sample(totuple(dneighbors((0, 0))), narms)
        succ = 0
        for armdir in armdirs:
            x, y = armdir
            arm = []
            for k in range(1, max(numh, numw)):
                nextloc = add(loc, (k * x * (oh + 1), k * y * (ow + 1)))
                if nextloc not in inds:
                    break
                arm.append(nextloc)
            if len(arm) < 2:
                continue
            aidx = unifint(diff_lb, diff_ub, (1, len(arm) - 1))
            endp = arm[aidx]
            gi = fill(gi, col, shift(sinds, endp))
            go = fill(go, col, mapply(lbind(shift, sinds), set(arm[:aidx+1])))
            succ += 1
        gi = fill(gi, col, shift(sinds, loc))
        go = fill(go, col, shift(sinds, loc))
    return {'input': gi, 'output': go}


def generate_90f3ed37(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (8, 30))
        w = unifint(diff_lb, diff_ub, (8, 30))
        pathh = unifint(diff_lb, diff_ub, (1, max(1, h//4)))
        pathh = unifint(diff_lb, diff_ub, (pathh, max(1, h//4)))
        Lpatper = unifint(diff_lb, diff_ub, (1, w//7))
        Rpatper = unifint(diff_lb, diff_ub, (1, w//7))
        hh = randint(1, pathh)
        Linds = asindices(canvas(-1, (hh, Lpatper)))
        Rinds = asindices(canvas(-1, (hh, Rpatper)))
        lpatsd = unifint(diff_lb, diff_ub, (0, (hh * Lpatper) // 2))
        rpatsd = unifint(diff_lb, diff_ub, (0, (hh * Rpatper) // 2))
        lpats = choice((lpatsd, hh * Lpatper - lpatsd))
        rpats = choice((rpatsd, hh * Rpatper - rpatsd))
        lpats = min(max(Lpatper, lpats), hh * Lpatper)
        rpats = min(max(Rpatper, rpats), hh * Rpatper)
        lpat = set(sample(totuple(Linds), lpats))
        rpat = set(sample(totuple(Rinds), rpats))
        midpatw = randint(0, w-2*Lpatper-2*Rpatper)
        if midpatw == 0 or Lpatper == hh == 1:
            midpat = set()
            midpatw = 0
        else:
            midpat = set(sample(totuple(asindices(canvas(-1, (hh, midpatw)))), randint(midpatw, (hh * midpatw))))
        if shift(midpat, (0, 2*Lpatper-midpatw)).issubset(lpat):
            midpat = set()
            midpatw = 0
        loci = randint(0, h - pathh)
        lplac = shift(lpat, (loci, 0)) | shift(lpat, (loci, Lpatper))
        mplac = shift(midpat, (loci, 2*Lpatper))
        rplac = shift(rpat, (loci, 2*Lpatper+midpatw)) | shift(rpat, (loci, 2*Lpatper+midpatw+Rpatper))
        sp = 2*Lpatper+midpatw+Rpatper
        for k in range(w//Lpatper+1):
            lplac |= shift(lpat, (loci, -k*Lpatper))
        for k in range(w//Rpatper+1):
            rplac |= shift(rpat, (loci, sp+k*Rpatper))
        pat = lplac | mplac | rplac
        patn = shift(pat, (-loci, 0))
        bgc, fgc = sample(cols, 2)
        gi = canvas(bgc, (h, w))
        gi = fill(gi, fgc, pat)
        options = interval(0, h - pathh + 1, 1)
        options = difference(options, interval(loci-pathh-1, loci+2*pathh, 1))
        nplacements = unifint(diff_lb, diff_ub, (1, max(1, len(options) // pathh)))
        go = tuple(e for e in gi)
        for k in range(nplacements):
            if len(options) == 0:
                break
            locii = choice(options)
            options = difference(options, interval(locii-pathh-1, locii+2*pathh, 1))
            hoffs = randint(0, max(Rpatper, w-sp-2))
            cutoffopts = interval(2*Lpatper+midpatw, 2*Lpatper+midpatw+hoffs+1, 1)
            cutoffopts = cutoffopts[::-1]
            idx = unifint(diff_lb, diff_ub, (0, len(cutoffopts) - 1))
            cutoff = cutoffopts[idx]
            patnc = sfilter(patn, lambda ij: ij[1] <= cutoff)
            go = fill(go, 1, shift(patn, (locii, hoffs)))
            gi = fill(gi, fgc, shift(patnc, (locii, hoffs)))
            go = fill(go, fgc, shift(patnc, (locii, hoffs)))
        if 1 in palette(go):
            break
    return {'input': gi, 'output': go}


def generate_36d67576(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        h = unifint(diff_lb, diff_ub, (10, 30))
        w = unifint(diff_lb, diff_ub, (10, 30))
        bgc, mainc, markerc = sample(cols, 3)
        remcols = difference(cols, (bgc, mainc, markerc))
        ncols = unifint(diff_lb, diff_ub, (1, len(remcols)))
        ccols = sample(remcols, ncols)
        gi = canvas(bgc, (h, w))
        oh = unifint(diff_lb, diff_ub, (2, 5))
        ow = unifint(diff_lb, diff_ub, (3 if oh == 2 else 2, 5))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = unifint(diff_lb, diff_ub, (4, len(bounds)))
        obj = {choice(totuple(bounds))}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        ntocompc = unifint(diff_lb, diff_ub, (1, ncells - 3))
        markercell = choice(totuple(obj))
        remobj = remove(markercell, obj)
        markercellobj = {(markerc, markercell)}
        tocompc = set(sample(totuple(remobj), ntocompc))
        mainpart = (obj - {markercell}) - tocompc
        mainpartobj = recolor(mainc, mainpart)
        tocompcobj = {(choice(remcols), ij) for ij in tocompc}
        obj = tocompcobj | mainpartobj | markercellobj
        smobj = mainpartobj | markercellobj
        smobjn = normalize(smobj)
        isfakesymm = False
        for symmf in [dmirror, cmirror, hmirror, vmirror]:
            if symmf(smobjn) == smobjn and symmf(obj) != obj:
                isfakesymm = True
                break
        if isfakesymm:
            continue
        loci = randint(0, h - oh)
        locj = randint(0, w - ow)
        plcd = shift(obj, (loci, locj))
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (asindices(gi) - plcdi) - mapply(neighbors, plcdi)
        noccs = unifint(diff_lb, diff_ub, (1, max(1, (h * w) // (2 * len(obj)))))
        succ = 0
        tr = 0
        maxtr = noccs * 5
        go = tuple(e for e in gi)
        while tr < maxtr and succ < noccs:
            tr += 1
            mf1 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf2 = choice((identity, dmirror, cmirror, hmirror, vmirror))
            mf = compose(mf1, mf2)
            outobj = normalize(mf(obj))
            inobj = sfilter(outobj, lambda cij: cij[0] in [mainc, markerc])
            oh, ow = shape(outobj)
            cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(cands) == 0:
                continue
            loc = choice(totuple(cands))
            outobjp = shift(outobj, loc)
            inobjp = shift(inobj, loc)
            outobjpi = toindices(outobjp)
            if outobjpi.issubset(inds):
                succ += 1
                inds = (inds - outobjpi) - mapply(neighbors, outobjpi)
                gi = paint(gi, inobjp)
                go = paint(go, outobjp)
        break
    return {'input': gi, 'output': go}


def generate_4522001f(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (3, 10))
    w = unifint(diff_lb, diff_ub, (3, 10))
    bgc, sqc, dotc = sample(cols, 3)
    gi = canvas(bgc, (h, w))
    go = canvas(bgc, (3*h, 3*w))
    sqi = {(dotc, (1, 1))} | recolor(sqc, {(0, 0), (0, 1), (1, 0)})
    sqo = backdrop(frozenset({(0, 0), (3, 3)}))
    sqo |= shift(sqo, (4, 4))
    loci = randint(0, min(h-2, 3*h-8))
    locj = randint(0, min(w-2, 3*w-8))
    loc = (loci, locj)
    plcdi = shift(sqi, loc)
    plcdo = shift(sqo, loc)
    gi = paint(gi, plcdi)
    go = fill(go, sqc, plcdo)
    noccs = unifint(diff_lb, diff_ub, (0, (h*w) // 9))
    succ = 0
    tr = 0
    maxtr = 10 * noccs
    iinds = ofcolor(gi, bgc) - mapply(dneighbors, toindices(plcdi))
    while tr < maxtr and succ < noccs:
        tr += 1
        cands = sfilter(iinds, lambda ij: ij[0] <= h - 2 and ij[1] <= w - 2)
        if len(cands) == 0:
            break
        loc = choice(totuple(cands))
        plcdi = shift(sqi, loc)
        plcdo = shift(sqo, loc)
        plcdii = toindices(plcdi)
        if plcdii.issubset(iinds):
            succ += 1
            iinds = (iinds - plcdii) - mapply(dneighbors, plcdii)
            gi = paint(gi, plcdi)
            go = fill(go, sqc, plcdo)
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_72322fa7(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    h = unifint(diff_lb, diff_ub, (10, 30))
    w = unifint(diff_lb, diff_ub, (10, 30))
    bgc = choice(cols)
    remcols = remove(bgc, cols)
    nobjs = unifint(diff_lb, diff_ub, (1, 4))
    ccols = sample(remcols, 2*nobjs)
    cpairs = list(zip(ccols[:nobjs], ccols[nobjs:]))
    objs = []
    gi = canvas(bgc, (h, w))
    inds = asindices(gi)
    for ca, cb in cpairs:
        oh = unifint(diff_lb, diff_ub, (1, 4))
        ow = unifint(diff_lb, diff_ub, (2 if oh == 1 else 1, 4))
        if choice((True, False)):
            oh, ow = ow, oh
        bounds = asindices(canvas(-1, (oh, ow)))
        obj = {choice(totuple(bounds))}
        ncells = randint(2, oh * ow)
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(neighbors, obj))))
        objn = normalize(obj)
        objt = totuple(objn)
        apart = sample(objt, randint(1, len(objt) - 1))
        bpart = difference(objt, apart)
        obj = recolor(ca, set(apart)) | recolor(cb, set(bpart))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: shift(objn, ij).issubset(inds))
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        gi = paint(gi, plcd)
        plcdi = toindices(plcd)
        inds = (inds - plcdi) - mapply(neighbors, plcdi)
        objs.append(obj)
    avgs = sum([len(o) for o in objs]) / len(objs)
    ub = max(1, (h * w) // (avgs * 2))
    noccs = unifint(diff_lb, diff_ub, (1, ub))
    succ = 0
    tr = 0
    maxtr = 5 * noccs
    go = tuple(e for e in gi)
    while tr < maxtr and succ < noccs:
        tr += 1
        obj = choice(objs)
        ca, cb = list(palette(obj))
        oh, ow = shape(obj)
        cands = sfilter(inds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
        if len(cands) == 0:
            continue
        loc = choice(totuple(cands))
        plcd = shift(obj, loc)
        plcdi = toindices(plcd)
        if plcdi.issubset(inds):
            succ += 1
            inds = (inds - plcdi) - mapply(neighbors, plcdi)
            go = paint(go, plcd)
            col = choice((ca, cb))
            gi = paint(gi, sfilter(plcd, lambda cij: cij[0] == col))
    return {'input': gi, 'output': go}


def generate_4290ef0e(diff_lb: float, diff_ub: float) -> dict:
    cols = interval(0, 10, 1)
    while True:
        d = unifint(diff_lb, diff_ub, (2, 7))
        h, w = d, d
        fullh = unifint(diff_lb, diff_ub, (4*d, 30))
        fullw = unifint(diff_lb, diff_ub, (4*d, 30))
        bgc = choice(cols)
        remcols = remove(bgc, cols)
        ccols = sample(remcols, d)
        quad = canvas(bgc, (d+1, d+1))
        for idx, c in enumerate(ccols):
            linlen = randint(2, w-idx+1)
            quad = fill(quad, c, (connect((idx, idx), (idx+linlen-1, idx))))
            quad = fill(quad, c, (connect((idx, idx), (idx, idx+linlen-1))))
        go = canvas(bgc, (d+1, 2*d+1))
        qobj1 = asobject(quad)
        qobj2 = shift(asobject(vmirror(quad)), (0, d))
        go = paint(go, qobj1)
        go = paint(go, qobj2)
        go = vconcat(go, hmirror(go)[1:])
        if choice((True, False)):
            go = fill(go, choice(difference(remcols, ccols)), {center(asindices(go))})
        objs = partition(go)
        objs = sfilter(objs, lambda o: color(o) != bgc)
        gi = canvas(bgc, (fullh, fullw))
        objs = order(objs, width)
        fullinds = asindices(gi)
        inds = asindices(gi)
        fullsuc = True
        for obj in objs:
            objn = normalize(obj)
            obji = toindices(objn)
            d = width(obj)
            dh = max(0, d//2-1)
            cands = sfilter(fullinds, lambda ij: ij[0] <= fullh - d and ij[1] <= fullw - d)
            cands = cands | shift(cands, (-dh, 0)) | shift(cands, (0, -dh)) | shift(cands, (dh, 0)) | shift(cands, (0, dh))
            maxtr = 10
            tr = 0
            succ = False
            if len(cands) == 0:
                break
            while tr < maxtr and not succ:
                tr += 1    
                loc = choice(totuple(cands))
                if (shift(obji, loc) & fullinds).issubset(inds):
                    succ = True
                    break
            if not succ:
                fullsuc = False
                break
            gi = paint(gi, shift(objn, loc))
            inds = inds - shift(obji, loc)
        if not fullsuc:
            continue
        break
    return {'input': gi, 'output': go}


def generate_6a1e5592(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(1, interval(0, 10, 1))
    h = unifint(diff_lb, diff_ub, (9, 30))
    w = unifint(diff_lb, diff_ub, (5, 30))
    barh = randint(3, h//3)
    maxobjh = h - barh - 1
    nobjs = unifint(diff_lb, diff_ub, (1, w//3))
    barc, bgc, objc = sample(cols, 3)
    c1 = canvas(barc, (barh, w))
    c2 = canvas(bgc, (h - barh, w))
    gi = vconcat(c1, c2)
    go = tuple(e for e in gi)
    tr = 0
    succ = 0
    maxtr = 10 * nobjs
    placopts = interval(1, w - 1, 1)
    iinds = ofcolor(gi, bgc)
    oinds = asindices(go)
    barinds = ofcolor(gi, barc)
    forbmarkers = set()
    while tr < maxtr and succ < nobjs:
        tr += 1
        oh = randint(1, maxobjh)
        ow = randint(1, min(4, w//2))
        bounds = asindices(canvas(-1, (oh, ow)))
        ncells = randint(1, oh * ow)
        sp = choice(totuple(connect((0, 0), (0, ow - 1))))
        obj = {sp}
        for k in range(ncells - 1):
            obj.add(choice(totuple((bounds - obj) & mapply(dneighbors, obj))))
        obj = normalize(obj)
        oh, ow = shape(obj)
        markerh = randint(1, min(oh, barh-1))
        markpart = sfilter(obj, lambda ij: ij[0] < markerh)
        markpartn = normalize(markpart)
        isinvalid = False
        for k in range(1, markerh+1):
            if normalize(sfilter(markpartn, lambda ij: ij[0] < k)) in forbmarkers:
                isinvalid = True
        if isinvalid:
            continue
        for k in range(1, markerh+1):
            forbmarkers.add(normalize(sfilter(markpartn, lambda ij: ij[0] < k)))
        placoptcands = sfilter(placopts, lambda jj: set(interval(jj, jj+ow+1, 1)).issubset(set(placopts)))
        if len(placoptcands) == 0:
            continue
        jloc = choice(placoptcands)
        iloc = barh - markerh
        oplcd = shift(obj, (iloc, jloc))
        if oplcd.issubset(oinds):
            icands = sfilter(iinds, lambda ij: ij[0] <= h - oh and ij[1] <= w - ow)
            if len(icands) == 0:
                continue
            loc = choice(totuple(icands))
            iplcd = shift(obj, loc)
            if iplcd.issubset(iinds):
                succ += 1
                iinds = (iinds - iplcd) - mapply(neighbors, iplcd)
                oinds = (oinds - oplcd)
                gi = fill(gi, objc, iplcd)
                gi = fill(gi, bgc, oplcd & barinds)
                go = fill(go, 1, oplcd)
                jm = apply(last, ofcolor(go, 1))
                placopts = sorted(difference(placopts, jm | apply(decrement, jm) | apply(increment, jm)))
        if len(placopts) == 0:
            break
    rotf = choice((identity, rot90, rot180, rot270))
    gi = rotf(gi)
    go = rotf(go)
    return {'input': gi, 'output': go}


def generate_e73095fd(diff_lb: float, diff_ub: float) -> dict:
    cols = remove(4, interval(0, 10, 1))
    while True:
        h = unifint(diff_lb, diff_ub, (10, 32))
        w = unifint(diff_lb, diff_ub, (10, 32))
        bgc, fgc = sample(cols, 2)
        gi = canvas(bgc, (h, w))
        nsplits = unifint(diff_lb, diff_ub, (2, min(h, w) // 3))
        for k in range(nsplits):
            objs = objects(gi, T, F, F)
            objs = colorfilter(objs, bgc)
            objs = apply(toindices, objs)
            hobjs = sfilter(objs, lambda o: height(o) > 6)
            wobjs = sfilter(objs, lambda o: width(o) > 6)
            if len(hobjs) == 0 and len(wobjs) == 0:
                break
            cgroups = [(g, ax) for g, ax in zip([hobjs, wobjs], [0, 1]) if len(g) > 0]
            g, ax = choice(cgroups)
            obj = choice(totuple(g))
            ulci, ulcj = ulcorner(obj)
            oh, ow = shape(obj)
            if ax == 0:
                iloc = randint(ulci + 3, ulci+oh-3)
                bar = sfilter(obj, lambda ij: ij[0] == iloc)
            else:
                jloc = randint(ulcj + 3, ulcj+ow-3)
                bar = sfilter(obj, lambda ij: ij[1] == jloc)
            gi = fill(gi, fgc, bar)
        copts = sfilter(
            ofcolor(gi, fgc),
            lambda ij: len(sfilter(toobject(dneighbors(ij), gi), lambda cij: cij[0] == fgc)) > 2
        )
        copts = sfilter(copts, lambda ij: len(sfilter(toobject(outbox(outbox({ij})), gi), lambda cij: cij[0] == fgc)) in {3, 4})
        if len(copts) == 0:
            continue
        noccs = unifint(diff_lb, diff_ub, (1, len(copts)))
        noccs = unifint(diff_lb, diff_ub, (noccs, len(copts)))
        occs = sample(totuple(copts), noccs)
        go = tuple(e for e in gi)
        forb = set()
        for occ in occs:
            ulci, ulcj = decrement(occ)
            lrci, lrcj = increment(occ)
            if len(sfilter(toobject(box({(ulci, ulcj), (lrci, lrcj)}), gi), lambda cij: cij[0] == fgc)) in {3, 4}:
                boptions = []
                for ulcioffs in [-2, -1, 0]:
                    for ulcjoffs in [-2, -1, 0]:
                        for lrcioffs in [0, 1, 2]:
                            for lrcjoffs in [0, 1, 2]:
                                bx = box({(ulci+ulcioffs, ulcj+ulcjoffs), (lrci+lrcioffs, lrcj+lrcjoffs)})
                                bxobj = toobject(bx, gi)
                                if len(sfilter(toobject(bxobj, gi), lambda cij: cij[0] == fgc)) in {3, 4} and len(sfilter(toobject(outbox(bxobj), gi), lambda cij: cij[0] == fgc)) in {3, 4}:
                                    boptions.append(bx)
                boptions = sfilter(boptions, lambda bx: len(backdrop(bx) & forb) == 0)
                if len(boptions) > 0:
                    bx = choice(boptions)
                    bd = backdrop(bx)
                    gi = fill(gi, bgc, bd)
                    gi = fill(gi, fgc, bx)
                    go = fill(go, 4, bd)
                    go = fill(go, fgc, bx)
                    forb |= bd
        gi = trim(gi)
        go = trim(go)
        if 4 in palette(go):
            break
    return {'input': gi, 'output': go}
