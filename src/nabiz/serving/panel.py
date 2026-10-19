"""Tek sayfalık izleme paneli. Toplayıcı `/` üzerinden sunar; dış varlık yok."""

PANEL_HTML = r"""<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NABIZ — alarmlar</title>
<style>
  :root {
    --bg:#0f172a; --card:#1b2336; --text:#f8fafc; --muted:#cbd5e1;
    --line:rgba(248,250,252,.16); --alarm:#fecaca; --sicak:#fde68a; --ok:#bbf7d0;
    --bar-a:#ef4444; --bar-s:#f59e0b; --bar-n:#22c55e; --ring:#f8fafc;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  body {
    margin:0; height:100vh; overflow:hidden; display:flex; flex-direction:column;
    font: 16px/1.5 "Segoe UI", system-ui, sans-serif; background:var(--bg); color:var(--text);
  }
  button, input { font: inherit; color: inherit; }
  button { cursor: pointer; }
  button:focus-visible, input:focus-visible, .kart:focus-visible {
    outline: 2px solid var(--ring); outline-offset: 2px;
  }
  header {
    flex:none; display:flex; justify-content:space-between; align-items:center; gap:16px;
    padding:12px 20px; border-bottom:1px solid var(--line);
  }
  h1 { font-size:18px; letter-spacing:.14em; font-weight:650; margin:0; }
  header p { margin:2px 0 0; color:var(--muted); max-width:62ch; }
  #durum { display:flex; align-items:center; gap:8px; color:var(--muted); text-align:right; }
  .nokta { width:8px; height:8px; border-radius:50%; background:var(--bar-n); flex:none; }
  .nokta.kapali { background:var(--bar-a); }
  main {
    flex:1; min-height:0; width:100%; padding:16px 20px;
    display:grid; grid-template-columns: minmax(260px, 340px) minmax(0, 1fr); gap:16px;
  }
  .makineler { display:flex; flex-direction:column; gap:12px; min-height:0; }
  .kart {
    flex:1; min-height:0; display:flex; flex-direction:column; gap:6px;
    background:var(--card); border:1px solid var(--line); border-radius:16px;
    padding:10px 10px 12px; text-align:left; transition: border-color 180ms ease, background 180ms ease;
  }
  .kart:hover { border-color: rgba(248,250,252,.4); }
  .kart[aria-pressed="true"] { border-color: var(--text); background:#222c42; }
  .kart canvas { flex:1; width:100%; min-height:96px; display:block; border-radius:10px; background:#121a2b; }
  .kart-yazi { display:flex; justify-content:space-between; align-items:flex-start; gap:8px; padding:0 4px; }
  .kart-yazi strong { display:block; font-size:18px; }
  .kart-yazi .kim { display:block; color:var(--muted); font-size:13px; }
  .kart-yazi .ozet { color:var(--muted); font-size:14px; text-align:right; }
  .not { margin:0; color:var(--muted); font-size:14px; }
  .calisma { min-height:0; display:flex; flex-direction:column; gap:12px; }
  .araclar { display:flex; flex-wrap:wrap; gap:12px 16px; align-items:flex-end; }
  .grup { display:flex; flex-direction:column; gap:6px; }
  .grup-ad { color:var(--muted); font-size:13px; }
  .seg { display:flex; flex-wrap:wrap; gap:8px; }
  .seg button, .ara {
    min-height:44px; padding:8px 14px; border-radius:999px;
    background:var(--card); border:1px solid var(--line); transition: border-color 180ms ease, background 180ms ease;
  }
  .seg button[aria-pressed="true"] { border-color: var(--text); background:#243044; }
  .ara { min-width:200px; }
  .sayilar { display:grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap:8px; }
  .sayi { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px 12px; }
  .sayi b { display:block; font-size:22px; line-height:1.15; font-variant-numeric: tabular-nums; }
  .sayi span { color:var(--muted); font-size:13px; }
  .dagilim { display:flex; height:28px; border-radius:8px; overflow:hidden; background:#243044; }
  .dagilim i {
    display:flex; align-items:center; justify-content:center; font-style:normal;
    font-size:13px; font-weight:650; color:#0f172a; min-width:0; overflow:hidden;
  }
  .dagilim .a { background:var(--bar-a); color:#fff; }
  .dagilim .s { background:var(--bar-s); }
  .dagilim .n { background:var(--bar-n); }
  .liste { flex:1; min-height:0; display:flex; flex-direction:column; gap:8px; }
  .tablo { flex:1; min-height:0; overflow:auto; border-radius:12px; background:var(--card); border:1px solid var(--line); }
  table { width:100%; border-collapse:collapse; table-layout:fixed; }
  caption { text-align:left; padding:10px 14px; color:var(--muted); }
  th, td { text-align:left; padding:10px 14px; border-bottom:1px solid var(--line); vertical-align:middle; }
  thead th { position:sticky; top:0; background:#1b2336; z-index:1; color:var(--text); font-weight:650; font-size:14px; }
  thead th span { display:block; font-weight:450; font-size:12px; color:var(--muted); }
  th:nth-child(1), td:nth-child(1) { width:16%; }
  th:nth-child(2), td:nth-child(2) { width:20%; white-space:nowrap; font-variant-numeric:tabular-nums; }
  th:nth-child(3), td:nth-child(3) { width:14%; }
  th:nth-child(4), td:nth-child(4) { width:16%; }
  td.alarm { color:var(--alarm); font-weight:650; }
  td.sicak { color:var(--sicak); font-weight:650; }
  td.soguk { color:var(--ok); }
  tr.alarm td:first-child { box-shadow: inset 3px 0 0 var(--bar-a); }
  tr.sicak td:first-child { box-shadow: inset 3px 0 0 var(--bar-s); }
  .bos td { color:var(--muted); box-shadow:none; }
  .anahtar { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:4px; }
  .anahtar li { display:flex; gap:8px; align-items:flex-start; color:var(--muted); font-size:14px; }
  .anahtar i { width:14px; height:14px; border-radius:3px; margin-top:3px; flex:none; }
  .anahtar b { color:var(--text); font-weight:650; }
  .anahtar .a { background:var(--bar-a); }
  .anahtar .s { background:var(--bar-s); }
  .anahtar .n { background:var(--bar-n); }
  .ayrim { margin:0; color:var(--muted); font-size:14px; }
  .cubuk { position:relative; height:10px; background:#243044; border-radius:6px; margin-top:6px; }
  .esik-not { display:block; color:var(--muted); font-size:12px; font-weight:450; }
  .dolu { position:absolute; left:0; top:0; bottom:0; border-radius:inherit; background:var(--bar-a); }
  .sinir { position:absolute; top:-3px; bottom:-3px; width:2px; background:#f8fafc; }
  .sayfalama { flex:none; display:flex; gap:8px; align-items:center; }
  .sayfalama button {
    min-height:44px; padding:8px 14px; border-radius:999px;
    background:var(--card); border:1px solid var(--line);
  }
  .sayfalama span { color:var(--muted); }
  button:disabled { opacity:.45; cursor:default; }
  @media (max-width: 980px) {
    body { height:auto; min-height:100vh; overflow:auto; }
    main { display:flex; flex-direction:column; }
    .makineler { flex-direction:row; flex-wrap:wrap; height:auto; }
    .kart { flex:1 1 220px; height:220px; }
    .not { flex-basis:100%; }
    .liste { min-height:420px; }
    .sayilar { grid-template-columns: 1fr 1fr; }
  }
  @media (max-width: 640px) {
    header, main { padding-left:12px; padding-right:12px; }
    .makineler { flex-direction:column; height:auto; }
    .kart { min-height:200px; }
    header p { display:none; }
  }
  @media (prefers-reduced-motion: reduce) {
    .kart, .seg button { transition: none; }
  }
</style>
</head>
<body>
<header>
  <div>
    <h1>NABIZ</h1>
    <p>Her satır bir ses dosyası. Kırmızı alarm verdi, turuncu şüpheli, yeşil normal saydı.</p>
  </div>
  <div id="durum"><span class="nokta" id="nokta"></span><span id="durum-yazi">bağlanıyor</span></div>
</header>
<main>
  <section class="makineler" id="makineler">
    <button type="button" class="kart" id="kart-fan" data-makine="fan" aria-pressed="false">
      <canvas id="c-fan" aria-hidden="true"></canvas>
      <div class="kart-yazi"><div><strong id="ad-fan">Fan</strong><span class="kim" id="kim-fan"></span></div><span class="ozet" id="ozet-fan">yükleniyor</span></div>
    </button>
    <button type="button" class="kart" id="kart-pump" data-makine="pump" aria-pressed="false">
      <canvas id="c-pump" aria-hidden="true"></canvas>
      <div class="kart-yazi"><div><strong id="ad-pump">Pompa</strong><span class="kim" id="kim-pump"></span></div><span class="ozet" id="ozet-pump">yükleniyor</span></div>
    </button>
    <p class="not">Karta tıkla: yalnız o makine. Aynı karta tekrar tıkla: ikisi birden.</p>
  </section>
  <section class="calisma">
    <div class="araclar" id="filtre">
      <div class="grup" role="group" aria-label="Klasör">
        <span class="grup-ad">Dosya nereden geldi</span>
        <div class="seg">
          <button type="button" data-k="klasor" data-v="" aria-pressed="true">hepsi</button>
          <button type="button" data-k="klasor" data-v="ariza">arıza klasörü</button>
          <button type="button" data-k="klasor" data-v="saglam">sağlam klasör</button>
        </div>
      </div>
      <div class="grup" role="group" aria-label="Sistem kararı">
        <span class="grup-ad">Sistem ne dedi</span>
        <div class="seg">
          <button type="button" data-k="karar" data-v="" aria-pressed="true">hepsi</button>
          <button type="button" data-k="karar" data-v="alarm">alarm verdi</button>
          <button type="button" data-k="karar" data-v="sicak">şüpheli</button>
          <button type="button" data-k="karar" data-v="soguk">normal saydı</button>
        </div>
      </div>
      <div class="grup">
        <label class="grup-ad" for="ara">Ses kaydının adı</label>
        <input class="ara" id="ara" type="search" placeholder="adın bir parçası" aria-label="dosya adı">
      </div>
    </div>
    <section class="sayilar" id="sayilar" aria-live="polite"></section>
    <div class="dagilim" id="dagilim" role="img" aria-label="sistem kararlarının dağılımı"></div>
    <ul class="anahtar">
      <li><i class="a"></i><span><b>Alarm verdi.</b> Sapma 2'yi geçen kayıtlar altı tane art arda geldi.</span></li>
      <li><i class="s"></i><span><b>Şüpheli.</b> Eşiği geçti ama art arda altıya ulaşmadı.</span></li>
      <li><i class="n"></i><span><b>Normal saydı.</b> Sapma 2'nin altında.</span></li>
    </ul>
    <p class="ayrim">Klasör ayrıdır: dosya arıza klasöründen mi geldi, sağlam klasöründen mi. Sistem kararı klasöre bakmaz.</p>
    <div class="liste">
      <div class="tablo">
        <table>
          <caption id="baslik">kayıtlar yükleniyor</caption>
          <thead><tr>
            <th>Ses kaydı<span>dosyanın adı</span></th>
            <th>Tarih ve saat<span>kaydın zamanı</span></th>
            <th>Makine<span>hangi cihaz</span></th>
            <th>Klasör<span>dosya nereden geldi</span></th>
            <th>Sistem ne dedi<span>bizim kararımız</span></th>
            <th>Sapma<span>beyaz çizgi eşik 2</span></th>
          </tr></thead>
          <tbody id="satirlar"></tbody>
        </table>
      </div>
      <div class="sayfalama">
        <button type="button" id="once">önceki</button>
        <span id="sayfa-yazi">sayfa</span>
        <button type="button" id="sonra">sonraki</button>
      </div>
    </div>
  </section>
</main>
<script>
var OLCEK = 16;
var SAYFA = 8;
var ham = [];
var sec = { makine: "", klasor: "", karar: "", ara: "" };
var sayfa = 0;
var hover = "";
var spin = 0;
var hareket = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, function (c) {
    return {"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c];
  });
}
function tarihYazi(ms) {
  var n = Number(ms);
  if (!n) return "—";
  var d = new Date(n);
  var secenek = { timeZone: "Europe/Istanbul" };
  var tarih = d.toLocaleDateString("tr-TR", Object.assign({ day: "2-digit", month: "2-digit", year: "numeric" }, secenek));
  var saat = d.toLocaleTimeString("tr-TR", Object.assign({ hour: "2-digit", minute: "2-digit", second: "2-digit" }, secenek));
  return tarih + " " + saat;
}
function makineAd(id) {
  var p = String(id || "").split("/");
  var tip = {fan:"Fan", pump:"Pompa"}[p[0]] || (p[0] || "Makine");
  var no = (p[1] || "").replace(/^id_/, "");
  return no ? tip + " " + no : tip;
}
function tur(s) {
  if (s.siddet === "ileri") return "alarm";
  if (s.sinif === "anomali") return "sicak";
  return "soguk";
}
function kararYazi(t) {
  if (t === "alarm") return "alarm verdi";
  if (t === "sicak") return "şüpheli";
  return "normal saydı";
}
function klasorYazi(k) {
  if (k === "ariza") return "arıza klasöründen";
  if (k === "saglam") return "sağlam klasöründen";
  return "klasör yok";
}
function uyar(s) {
  if (sec.makine && String(s.makine_id).indexOf(sec.makine + "/") !== 0) return false;
  if (sec.klasor && s.dugum_id !== sec.klasor) return false;
  if (sec.karar && tur(s) !== sec.karar) return false;
  if (sec.ara && String(s.kayit_adi || "").toLowerCase().indexOf(sec.ara) < 0) return false;
  return true;
}
function derece(s) {
  var t = tur(s);
  return t === "alarm" ? 0 : t === "sicak" ? 1 : 2;
}
function liste() {
  return ham.filter(uyar).slice().sort(function (a, b) {
    var d = derece(a) - derece(b);
    if (d) return d;
    return Number(b.anomali_skoru) - Number(a.anomali_skoru);
  });
}
function birimKim(tip) {
  var gor = {};
  ham.forEach(function (s) {
    if (String(s.makine_id).indexOf(tip + "/") === 0) gor[s.makine_id] = true;
  });
  var liste = Object.keys(gor).sort();
  if (!liste.length) return { baslik: tip === "fan" ? "Fan" : "Pompa", kim: "birim yok" };
  return {
    baslik: liste.map(makineAd).join(", "),
    kim: (tip === "pump" ? "santrifüj pompa · " : "eksenel fan · ") +
      liste.map(function (id) { return id.replace("/", " · "); }).join(", ")
  };
}
function makineOzet(tip) {
  var L = ham.filter(function (s) { return String(s.makine_id).indexOf(tip + "/") === 0; });
  var alarm = L.filter(function (s) { return tur(s) === "alarm"; }).length;
  if (!L.length) return "ses kaydı yok";
  if (alarm) return alarm + " alarm verdi · " + L.length + " ses kaydı";
  return "alarm yok · " + L.length + " ses kaydı";
}
function boya() {
  document.querySelectorAll("#filtre button").forEach(function (b) {
    var acik = (sec[b.dataset.k] || "") === b.dataset.v;
    b.setAttribute("aria-pressed", acik ? "true" : "false");
  });
  ["fan", "pump"].forEach(function (tip) {
    var kart = document.getElementById("kart-" + tip);
    kart.setAttribute("aria-pressed", sec.makine === tip ? "true" : "false");
    var kim = birimKim(tip);
    document.getElementById("ad-" + tip).textContent = kim.baslik;
    document.getElementById("kim-" + tip).textContent = kim.kim;
    document.getElementById("ozet-" + tip).textContent = makineOzet(tip);
  });
}

function yuz(pts, renk) {
  var z = 0;
  for (var i = 0; i < pts.length; i++) z += pts[i][2];
  return { pts: pts, renk: renk, z: z / pts.length };
}
function don(p, spinA, tilt) {
  var c = Math.cos(spinA), s = Math.sin(spinA);
  var x = p[0] * c - p[1] * s, y = p[0] * s + p[1] * c;
  var ct = Math.cos(tilt), st = Math.sin(tilt);
  return [x, y * ct - p[2] * st, y * st + p[2] * ct];
}
function halka(r0, r1, z0, z1, n, renk, spinA, tilt) {
  var out = [];
  for (var i = 0; i < n; i++) {
    var a0 = i / n * Math.PI * 2, a1 = (i + 1) / n * Math.PI * 2;
    out.push(yuz([
      don([Math.cos(a0)*r0, Math.sin(a0)*r0, z0], spinA, tilt),
      don([Math.cos(a1)*r0, Math.sin(a1)*r0, z0], spinA, tilt),
      don([Math.cos(a1)*r1, Math.sin(a1)*r1, z1], spinA, tilt),
      don([Math.cos(a0)*r1, Math.sin(a0)*r1, z1], spinA, tilt)
    ], renk));
  }
  return out;
}
function kanat(spinA, tilt, renk) {
  var out = [];
  for (var b = 0; b < 6; b++) {
    var a = b / 6 * Math.PI * 2;
    out.push(yuz([
      don([Math.cos(a-0.08)*0.22, Math.sin(a-0.08)*0.22, 0.06], spinA, tilt),
      don([Math.cos(a+0.16)*0.22, Math.sin(a+0.16)*0.22, 0.06], spinA, tilt),
      don([Math.cos(a+0.34)*1.42, Math.sin(a+0.34)*1.42, 0.0], spinA, tilt),
      don([Math.cos(a-0.02)*1.42, Math.sin(a-0.02)*1.42, 0.0], spinA, tilt)
    ], renk));
  }
  return out;
}
function prizma(x0, x1, y0, y1, z0, z1, renkler, tilt) {
  function P(x, y, z) { return don([x, y, z], 0, tilt); }
  return [
    yuz([P(x0, y0, z0), P(x1, y0, z0), P(x1, y1, z0), P(x0, y1, z0)], renkler[0]),
    yuz([P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], renkler[1]),
    yuz([P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)], renkler[2]),
    yuz([P(x0, y0, z0), P(x1, y0, z0), P(x1, y0, z1), P(x0, y0, z1)], renkler[3]),
    yuz([P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)], renkler[4])
  ];
}
function silindirX(x0, x1, r, tonlar, tilt) {
  var n = tonlar.length, out = [];
  for (var i = 0; i < n; i++) {
    var a0 = -Math.PI / 2 + i / n * Math.PI * 2;
    var a1 = -Math.PI / 2 + (i + 1) / n * Math.PI * 2;
    out.push(yuz([
      don([x0, Math.cos(a0) * r, Math.sin(a0) * r], 0, tilt),
      don([x1, Math.cos(a0) * r, Math.sin(a0) * r], 0, tilt),
      don([x1, Math.cos(a1) * r, Math.sin(a1) * r], 0, tilt),
      don([x0, Math.cos(a1) * r, Math.sin(a1) * r], 0, tilt)
    ], tonlar[i]));
  }
  return out;
}
function kapakX(x, r, renk, tilt) {
  var n = 14, c = don([x, 0, 0], 0, tilt), out = [];
  for (var i = 0; i < n; i++) {
    var a0 = i / n * Math.PI * 2, a1 = (i + 1) / n * Math.PI * 2;
    out.push(yuz([
      c,
      don([x, Math.cos(a0) * r, Math.sin(a0) * r], 0, tilt),
      don([x, Math.cos(a1) * r, Math.sin(a1) * r], 0, tilt)
    ], renk));
  }
  return out;
}
function disk(r, z, renk, tilt, n) {
  var c = don([0, 0, z], 0, tilt), out = [];
  for (var i = 0; i < n; i++) {
    var a0 = i / n * Math.PI * 2, a1 = (i + 1) / n * Math.PI * 2;
    out.push(yuz([
      c,
      don([Math.cos(a0) * r, Math.sin(a0) * r, z], 0, tilt),
      don([Math.cos(a1) * r, Math.sin(a1) * r, z], 0, tilt)
    ], renk));
  }
  return out;
}
function salyangoz(tilt) {
  var out = [], n = 42;
  for (var i = 0; i < n; i++) {
    var t0 = i / n, t1 = (i + 1) / n;
    var a0 = -Math.PI / 2 - (1 - t0) * Math.PI * 2;
    var a1 = -Math.PI / 2 - (1 - t1) * Math.PI * 2;
    var r0 = 0.56 + 0.56 * t0, r1 = 0.56 + 0.56 * t1;
    var govde = t0 > 0.78 ? "#2f7fa3" : "#184861";
    out.push(yuz([
      don([0, 0, 0.02], 0, tilt),
      don([Math.cos(a0) * r0, Math.sin(a0) * r0, 0.02], 0, tilt),
      don([Math.cos(a1) * r1, Math.sin(a1) * r1, 0.02], 0, tilt)
    ], govde));
    out.push(yuz([
      don([Math.cos(a0) * r0, Math.sin(a0) * r0, -0.1], 0, tilt),
      don([Math.cos(a1) * r1, Math.sin(a1) * r1, -0.1], 0, tilt),
      don([Math.cos(a1) * (r1 + 0.13), Math.sin(a1) * (r1 + 0.13), 0.12], 0, tilt),
      don([Math.cos(a0) * (r0 + 0.13), Math.sin(a0) * (r0 + 0.13), 0.12], 0, tilt)
    ], t0 > 0.78 ? "#5aa4c4" : "#12384c"));
  }
  return out;
}
function cark(spinA, tilt, renk) {
  var out = [];
  for (var b = 0; b < 6; b++) {
    var a = b / 6 * Math.PI * 2;
    out.push(yuz([
      don([Math.cos(a) * 0.05, Math.sin(a) * 0.05, 0.28], spinA, tilt),
      don([Math.cos(a + 0.16) * 0.05, Math.sin(a + 0.16) * 0.05, 0.28], spinA, tilt),
      don([Math.cos(a + 0.62) * 0.2, Math.sin(a + 0.62) * 0.2, 0.26], spinA, tilt),
      don([Math.cos(a + 0.3) * 0.2, Math.sin(a + 0.3) * 0.2, 0.26], spinA, tilt)
    ], renk));
  }
  return out;
}
function basma(tilt) {
  var boyun = ["#8eabba", "#f7fcfe", "#d7e8ef", "#6e909f", "#a9c5d1"];
  var flans = ["#5c7c8c", "#d5e4ea", "#b7d0da", "#4c6a78", "#8eacba"];
  return prizma(-0.32, 0.32, -1.92, -0.86, -0.1, 0.24, boyun, tilt)
    .concat(prizma(-0.56, 0.56, -2.18, -1.82, -0.2, 0.34, flans, tilt));
}
function pompa(spinA, metal) {
  var tilt = -0.42;
  var celik = ["#e8eef1", "#c9d3d8", "#93a2a8", "#62727a", "#3e4c54", "#2b363c", "#263238", "#45545c", "#74848c", "#bcc8cd"];
  var kaburga = ["#dfe7eb", "#b7c3c9", "#84949b", "#56666e", "#36444b", "#243036", "#202a30", "#3d4c54", "#687880", "#aebbc2"];
  var kaplin = ["#1c2830", "#314048", "#4d616b", "#6d828c", "#8ea3ac", "#6d828c", "#4d616b", "#314048", "#283840", "#1c2830"];
  var ayak = ["#142430", "#243848", "#1c3040", "#101c28", "#1a2c3a"];
  var faces = kapakX(-2.28, 0.4, "#222c32", tilt)
    .concat(silindirX(-2.28, -0.92, 0.4, celik, tilt));
  [-2.1, -1.78, -1.46, -1.14].forEach(function (x) {
    faces = faces.concat(silindirX(x, x + 0.08, 0.52, kaburga, tilt));
  });
  faces = faces.concat(silindirX(-0.92, -0.7, 0.18, kaplin, tilt))
    .concat(salyangoz(tilt))
    .concat(prizma(-2.2, 1.15, 1.02, 1.2, -0.28, 0.18, ayak, tilt))
    .concat(prizma(-1.85, -1.45, 1.2, 1.5, -0.16, 0.1, ayak, tilt))
    .concat(prizma(0.45, 0.85, 1.2, 1.5, -0.16, 0.1, ayak, tilt));
  var onde = disk(0.3, 0.22, "#0c1822", tilt, 22)
    .concat(cark(spinA, tilt, metal))
    .concat(disk(0.07, 0.32, "#d7c09a", tilt, 10));
  return { faces: faces, onde: onde, cikis: basma(tilt) };
}
function parcalar(id, spinA, metal) {
  if (id === "pump") return pompa(spinA, metal);
  var tilt = -0.95;
  var faces = halka(1.5, 1.78, -0.12, 0.16, 22, "#1f6f62", 0, tilt)
    .concat(kanat(spinA, tilt, metal))
    .concat(halka(0, 0.28, 0.02, 0.2, 8, "#d5b48a", spinA, tilt));
  return { faces: faces, cikis: [] };
}
function sinir(parca) {
  var minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  parca.faces.concat(parca.onde || []).concat(parca.cikis).forEach(function (f) {
    f.pts.forEach(function (p) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
    });
  });
  return { minX: minX, maxX: maxX, minY: minY, maxY: maxY };
}
var olcu = null;
function olculer() {
  if (olcu) return olcu;
  olcu = { fan: sinir(parcalar("fan", 0, "#b7d5cc")), pump: sinir(parcalar("pump", 0, "#b7d5cc")) };
  return olcu;
}
function cizModel(ctx, cx, cy, id, secili, ustunde, olcek) {
  var spinA = spin * (id === "fan" ? 1.4 : 0.7);
  var metal = secili || ustunde ? "#f3fff8" : "#d5e4ea";
  var parc = parcalar(id, spinA, metal);
  var faces = parc.faces;
  var onde = parc.onde || [];
  var cikis = parc.cikis;
  faces.sort(function (a, b) { return a.z - b.z; });
  function cizYuz(f, kenar) {
    ctx.beginPath();
    f.pts.forEach(function (p, i) {
      var x = cx + p[0] * olcek, y = cy + p[1] * olcek;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = f.renk;
    ctx.fill();
    if (kenar) { ctx.strokeStyle = "rgba(8, 16, 28, 0.45)"; ctx.lineWidth = 1; ctx.stroke(); }
  }
  faces.forEach(function (f) { cizYuz(f, false); });
  onde.forEach(function (f) { cizYuz(f, false); });
  cikis.forEach(function (f) { cizYuz(f, true); });
}
function cizTuval(id) {
  var canvas = document.getElementById(id === "fan" ? "c-fan" : "c-pump");
  var ctx = canvas.getContext("2d");
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  var w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  var bw = Math.floor(w * dpr), bh = Math.floor(h * dpr);
  if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  var s = olculer()[id === "fan" ? "fan" : "pump"];
  var pad = 12;
  var en = s.maxX - s.minX, yuk = s.maxY - s.minY;
  var olcek = Math.min((w - pad * 2) / en, (h - pad * 2) / yuk);
  if (!isFinite(olcek) || olcek <= 0) return;
  var cx = (w - (s.minX + s.maxX) * olcek) / 2;
  var cy = (h - (s.minY + s.maxY) * olcek) / 2;
  cizModel(ctx, cx, cy, id, sec.makine === id, hover === id, olcek);
}

function cubuk(skor) {
  var s = Math.max(0, Number(skor));
  var gen = s <= 2 ? (s / 2) * 32 : 32 + Math.min(68, (Math.log(s / 2) / Math.log(40)) * 68);
  gen = Math.max(0, Math.min(100, gen));
  var not = s >= 2 ? "eşik 2'nin üstünde" : "eşik 2'nin altında";
  return '<div class="cubuk" role="img" aria-label="sapma ' + s.toFixed(1) + ', ' + not + '"><div class="dolu" style="width:' +
    gen.toFixed(1) + '%"></div><i class="sinir" style="left:32%"></i></div><span class="esik-not">' + not + "</span>";
}
function satirSigar() {
  var kasa = document.querySelector(".tablo");
  if (!kasa || kasa.clientHeight < 120) return SAYFA;
  var n = Math.floor((kasa.clientHeight - 78) / 58);
  return Math.max(6, Math.min(20, n));
}
var tabloDamga = "";
function cizTablo() {
  var L = liste();
  var sigar = satirSigar();
  if (sigar !== SAYFA) {
    var imlec = sayfa * SAYFA;
    SAYFA = sigar;
    sayfa = Math.floor(imlec / SAYFA);
  }
  var sayfaSay = Math.max(1, Math.ceil(L.length / SAYFA) || 1);
  if (sayfa >= sayfaSay) sayfa = sayfaSay - 1;
  if (sayfa < 0) sayfa = 0;
  var damga = [L.length, ham.length, sec.makine, sec.klasor, sec.karar, sec.ara, sayfa, SAYFA].join("|");
  if (damga === tabloDamga) { boya(); return; }
  tabloDamga = damga;
  var alarm = L.filter(function (s) { return tur(s) === "alarm"; }).length;
  var sicak = L.filter(function (s) { return tur(s) === "sicak"; }).length;
  var soguk = L.length - alarm - sicak;
  var ariza = L.filter(function (s) { return s.dugum_id === "ariza"; }).length;
  var arizaYakalanan = L.filter(function (s) { return s.dugum_id === "ariza" && tur(s) !== "soguk"; }).length;
  document.getElementById("sayilar").innerHTML =
    kutu(L.length, "görünen ses kaydı / " + ham.length) +
    kutu(alarm, "alarm verdi") +
    kutu(sicak, "şüpheli, seri tamam değil") +
    kutu(ariza ? Math.round(100 * arizaYakalanan / ariza) + "%" : "–", "arıza klasöründekilerin kaçı eşiği geçti");
  document.getElementById("dagilim").innerHTML =
    dilim("a", alarm, L.length) + dilim("s", sicak, L.length) + dilim("n", soguk, L.length);
  var bas = sayfa * SAYFA;
  var dil = L.slice(bas, bas + SAYFA);
  document.getElementById("baslik").textContent = L.length
    ? (bas + 1) + "–" + (bas + dil.length) + " / " + L.length + " ses kaydı. Önce alarm verdiklerimiz."
    : "Bu süzgeçte ses kaydı yok.";
  document.getElementById("sayfa-yazi").textContent = "sayfa " + (sayfa + 1) + " / " + sayfaSay;
  document.getElementById("once").disabled = sayfa <= 0;
  document.getElementById("sonra").disabled = sayfa >= sayfaSay - 1;
  document.getElementById("satirlar").innerHTML = dil.length ? dil.map(function (s) {
    var t = tur(s);
    return "<tr class=\"" + t + "\"><td>" + esc(s.kayit_adi || "—") + "</td><td>" + esc(tarihYazi(s.zaman_damgasi_ms)) +
      "</td><td>" + esc(makineAd(s.makine_id)) +
      "</td><td>" + esc(klasorYazi(s.dugum_id)) + "</td><td class=\"" + t + "\">" + esc(kararYazi(t)) +
      "</td><td><b>" + Number(s.anomali_skoru).toFixed(1) + "</b>" + cubuk(s.anomali_skoru) + "</td></tr>";
  }).join("") : "<tr class=\"bos\"><td colspan=\"6\">Bu süzgeçte ses kaydı yok. «hepsi»ye bas.</td></tr>";
  boya();
}
function kutu(n, yazi) { return '<article class="sayi"><b>' + esc(n) + '</b><span>' + esc(yazi) + "</span></article>"; }
function dilim(sin, n, top) {
  if (!top || !n) return "";
  var y = (100 * n / top);
  return '<i class="' + sin + '" style="width:' + y + '%">' + (y >= 16 ? "%" + Math.round(y) : "") + "</i>";
}

async function yenile() {
  var nokta = document.getElementById("nokta");
  var yazi = document.getElementById("durum-yazi");
  try {
    var saglik = await fetch("/saglik").then(function (r) { return r.json(); });
    ham = await fetch("/tespit?limit=20000").then(function (r) { return r.json(); });
    nokta.classList.remove("kapali");
    var son = ham.reduce(function (m, s) { return Math.max(m, Number(s.zaman_damgasi_ms) || 0); }, 0);
    yazi.textContent = "canlı · " + saglik.toplam_kayit + " kayıt · son kayıt " + tarihYazi(son);
    cizTablo();
    requestAnimationFrame(function () { cizTablo(); });
  } catch (e) {
    nokta.classList.add("kapali");
    yazi.textContent = "sunucuya ulaşılamadı";
  }
}
document.getElementById("filtre").addEventListener("click", function (ev) {
  var b = ev.target.closest("button");
  if (!b) return;
  sec[b.dataset.k] = b.dataset.v;
  sayfa = 0;
  cizTablo();
});
document.getElementById("ara").addEventListener("input", function (ev) {
  sec.ara = ev.target.value.trim().toLowerCase();
  sayfa = 0;
  cizTablo();
});
document.getElementById("once").addEventListener("click", function () { if (sayfa > 0) { sayfa--; cizTablo(); } });
document.getElementById("sonra").addEventListener("click", function () { sayfa++; cizTablo(); });
document.getElementById("makineler").addEventListener("click", function (ev) {
  var b = ev.target.closest("[data-makine]");
  if (!b) return;
  sec.makine = sec.makine === b.dataset.makine ? "" : b.dataset.makine;
  sayfa = 0;
  cizTablo();
});
document.getElementById("makineler").addEventListener("mousemove", function (ev) {
  var b = ev.target.closest("[data-makine]");
  hover = b ? b.dataset.makine : "";
});
document.getElementById("makineler").addEventListener("mouseleave", function () { hover = ""; });
window.addEventListener("resize", function () {
  var n = satirSigar();
  if (n !== SAYFA) {
    var imlec = sayfa * SAYFA;
    SAYFA = n;
    sayfa = Math.floor(imlec / SAYFA);
    cizTablo();
  }
});
function kare() {
  if (hareket) spin += 0.012;
  cizTuval("fan");
  cizTuval("pump");
  requestAnimationFrame(kare);
}
kare();
yenile();
setInterval(yenile, 4000);
</script>
</body>
</html>
"""
