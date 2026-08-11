# LeKiwi Physical Robot — Bill of Materials

Final component list for Phase 9, with direct checkout links and verified stock status as of **August 2026**.

**Cost-reduction pass (2026-08-10):** re-priced every component *except* the Seeed LeKiwi Kit and Seeed X10 camera against the least expensive still-legitimate option findable (no counterfeits/clone listings, no unverifiable marketplace sellers) — target was a **$300–350 total build cost**. Real savings were found (RPLIDAR down $20.98, Raspberry Pi down $60–65 by buying **8GB in-store at Micro Center**, a real option since there's one in Marietta, GA), but **the honest total still lands at ~$437–455, not $300–350** — see "Why $300–350 isn't reachable right now" below before ordering anything on the assumption the target was hit.

**Camera swap (2026-08-11):** the Seeed X10 was replaced with an **Arducam IMX291 USB2.0 board camera (SKU B0200, "100° Wide Angle")** — real, in-stock, UVC/plug-and-play, still 1080p. Reason: the X10's real FOV was never published anywhere (Seeed's product page, a reseller, their wiki, and the actual datasheet PDF all lack it — confirmed by direct fetch, not a search gap), and the simulator's own optics assumption for it was a ~53.5° placeholder that was genuinely too narrow for a nav task. The Arducam board has a real, manufacturer-published ~100°(D) FOV (Arducam's own product-family sibling datasheet, B0201, confirmed by direct PDF fetch) — much closer to the simulator's newly-widened 90° camera FOV and the lidar's newly-narrowed 90° FOV (see `plan.md`/`plan_log.md` Phase 3/6) than the X10 ever was. **Price could not be tool-verified this session** — every retailer selling this board (Arducam's own store, UCTronics, RobotShop) returned HTTP 403 to automated fetches, and Amazon's price is JavaScript-rendered and invisible to the fetch tool used here. Confirm the real price at checkout before ordering — flagged honestly below, not guessed at.

---

## Core Components

| Component | SKU / Model | Qty | Price (USD) | Stock Status | Link |
|-----------|-----------|-----|-----------|--------------|------|
| **Seeed LeKiwi Kit 12V** (mobile base, 3D printed parts, battery) — *unchanged, excluded from this cost pass* | LeKiwi-12V | 1 | $179.00 | ⚠️ Pre-order / Limited | [Seeed Studio (JP)](https://jp.seeedstudio.com/mobile-base-c-2676.html) |
| **Arducam IMX291 USB2.0 Camera Module, 100° FOV** (front-facing RGB, replaces the Seeed X10 — see camera-swap note above) | B0200 | 1 | ⚠️ **Unconfirmed** — every retailer checked (Arducam.com, UCTronics, RobotShop) blocks automated price fetches; Amazon's price is JS-rendered and invisible to the fetch tool used this session | ⚠️ Availability not tool-confirmable either, for the same reason — widely listed across multiple retailers, which is a good sign, but check stock at checkout | [Amazon](https://www.amazon.com/Arducam-Camera-Computer-Microphone-Windows/dp/B07ZRJDTBQ) · [RobotShop](https://www.robotshop.com/products/arducam-1080p-usb2-uvc-mini-camera-microphone-2mp-128in-cmos-imx291-100) · [Arducam](https://www.arducam.com/arducam-1080p-low-light-wdr-usb-camera-module-for-computer-2mp-1-2-8-cmos-imx291-100-degree-wide-angle-mini-uvc-spy-webcam-board-with-microphone-3-3ft-1m-cable-for-windows-linux-mac-os.html) |
| **Raspberry Pi 5 8GB** (barebone board) | RPI5-8GB | 1 | $114.99 | ⚠️ In-store only, Micro Center — **Marietta, GA confirmed accessible, live stock at that specific store not confirmable by automated fetch (microcenter.com blocks it, HTTP 403)** — check the site's store selector or call ahead before driving over | [Micro Center](https://www.microcenter.com/product/702589/raspberry-pi-5-8gb) |
| **RPLIDAR A1M8-R6 360° LiDAR** (360° 2D scanner, 12m range) | A1M8-R6 | 1 | $99.00 | ✅ In stock | [Seeed Studio](https://www.seeedstudio.com/RPLiDAR-A1M8-R6-360-Degree-Laser-Scanner-Kit-12M-Range-p-4785.html) |

**Total Hardware Cost (excluding shipping, taxes, mount hardware below):** ~$393.00 + the unconfirmed camera price (was ~$405.98 with the $12.99 X10; the Arducam board's real price still needs to be checked at checkout before this total means anything).

### What changed and why

- **RPLIDAR: $119.98 → $99.00** (Walmart/LYK-Technology reseller → Seeed Studio direct). Real, confirmed floor price — cross-checked against Slamtec's own official AliExpress store, which also lists $99.00 for the same part, so this isn't a fluke listing, it's the genuine current price for a real (non-clone) A1M8. RobotShop's older $103.12 figure (cited in the previous pass) is no longer the cheapest verified option.
- **Raspberry Pi 5: $175–200 → $114.99**, by buying **8GB in-store at Micro Center** instead of online. Confirmed there's a Micro Center in Marietta, GA — real, driveable option, not hypothetical. Every 8GB reseller checked *online* this pass (PiShop.us, CanaKit, SparkFun) is still clustered at **$175–180**, not the $95 official MSRP — a real, industry-wide condition (Raspberry Pi's own blog has publicly acknowledged 2025–2026 memory-cost-driven price increases on every 2GB+ SKU, calling it "painful but ultimately temporary"), not a search failure. Micro Center's $114.99 beats every online 8GB price by $60+ **and** keeps the full 8GB (no RAM tradeoff, unlike the 4GB fallback below).
- **One real gap in this recommendation, said plainly:** microcenter.com returns HTTP 403 to automated fetches, so the $114.99 price and in-stock status above are confirmed for Micro Center generally (via search-indexed cache), not verified live for the specific Marietta store's current inventory. Micro Center's own online stock checker (by store) or a phone call before driving over is the real way to confirm before making the trip.

### Fallback if Marietta doesn't have stock: online, 4GB, no store trip

**Raspberry Pi 5 4GB** — $110.00 @ [PiShop.us](https://www.pishop.us/product/raspberry-pi-5-4gb/) (official reseller, in stock, ships). Real tradeoff, not free: this project's actual on-device workload is running a *trained* policy (inference only, no on-device training — see `plan.md` Phase 10), so 4GB is very likely sufficient, but hasn't been load-tested against the real LeRobot control stack. Only ~$5 cheaper than Micro Center's 8GB, so Micro Center is the better deal whenever it's actually in stock — this is purely the "don't want to drive to Marietta, or they're out of stock" option.

---

## Lidar Mount Hardware (for the 3D-printed RPLIDAR A1M8 mount, `lidar_mount_block_v1.stl`)

The mount design uses **two different screw sizes**, not just M3 — worth knowing before
ordering, since it's easy to assume one size covers the whole assembly:

- **M3** for the 4 screws holding the mount block down to the robot's own top base
  plate (matches the plate's real 3.5mm hole grid).
- **M2.5**, not M3, for the 2 screws holding the RPLIDAR A1M8 itself to the block —
  that's Slamtec's own stock hole size on the real unit (confirmed in their datasheet),
  not a choice made for this build.

**Cheaper approach this pass: one combined insert+screw+tip kit instead of three separate purchases.** The previous BoM bought M3 inserts, M2.5 inserts, and a soldering-iron tip kit as three separate line items ($10.99 + $11.99 + $32.40 = $55.38 before screws). Combined small-quantity kits that bundle both insert sizes *and* the matching soldering-iron tips exist (e.g. search "M2.5 M3 threaded insert soldering iron tip kit" on Amazon — several listings in the $15–25 range bundle exactly the two sizes this build needs plus tips, since this project only needs 6 inserts total, not a 100-piece pack of each size). **Not linked directly below** — Amazon product pricing isn't reliably fetchable by automated tools (confirmed again this pass, same limitation the previous BoM already flagged), so treat the estimate as a real, findable range, not a locked-in number; check current listings and pick one compatible with whatever soldering iron you already have (TS100/TS101/Pinecil-style irons are the most common/cheapest tip-compatibility class).

| Component | Qty needed | Price (USD) | Notes |
|-----------|-----------|-----------|------|
| **Combined M2.5+M3 heat-set insert kit with soldering-iron tips** (small quantity, ~10-20pc, not a 100-pack of each) | 1 kit | ~$15–25 | Replaces the three separate purchases below; search current listings, exact price not automatable |
| **M3 x 12mm socket head cap screws** (4 needed, buy an assorted-length kit) | 4 | ~$8–12 | ⚠️ Price not confirmable via automated fetch — check at checkout |
| **M2.5 x 8mm socket head cap screws** (2 needed, buy an assorted-length kit) | 2 | ~$8–12 | ⚠️ Price not confirmable via automated fetch — check at checkout |

**Fallback (previous pass's separate-purchase approach, if a combined kit isn't easily found in stock):**
M3 heat-set brass inserts (100-pc, only need 4) — $10.99 @ [Prusa3D](https://www.prusa3d.com/product/heat-set-inserts-m3-standard-100-pcs/); M2.5 heat-set brass inserts (100-pc, only need 2) — $11.99 @ [Prusa3D](https://www.prusa3d.com/product/threaded-inserts-m2-5-standard-100-pcs/); heat-set insert tip kit (TS100/TS101/Pinecil-compatible) — $32.40 @ [kb-3d.com](https://kb-3d.com/store/tools-equipment/976-cnckitchen-heat-set-insert-tool-kit-ts100ts101-1688693301779.html).

**Not a heat gun** — heat-set inserts need a **soldering iron** (controlled temperature,
firm even downward pressure to seat the insert square), not a heat gun; a heat gun can't
apply that directional force and tends to overheat/warp the surrounding plastic.

**Estimated added cost:** ~$31–49 with the combined-kit approach (down from the previous pass's ~$55–75), or ~$55–75 via the separate-purchase fallback if a combined kit isn't available when you actually order.

---

## Camera Mount Hardware (for the 3D-printed Arducam bracket, `camera_mount_bracket_v1.stl`)

New this pass, built the same way as the lidar mount above: a real STL generated against the base plate's actual measured hole grid (`urdf/meshes/camera_base_base_plate_layer1_v5.stl`, cross-sectioned directly, not assumed) and the Arducam board's own published mechanical drawing (see `plan_log.md`'s camera-swap section for the full sourcing, including the one real caveat: the drawing used is from the B0201 sibling SKU, not B0200's own datasheet specifically — same board family, same PCB, different lens, per Arducam's own product-family convention).

**Also two screw sizes, same reasoning as the lidar mount:**
- **M3** — 6 screws, base flange to the real base-plate holes (measured ~1.727mm hole radius on the actual plate mesh, i.e. a real M3 clearance hole, not picked to match an assumed screw size).
- **M2.5** — 4 screws, Arducam board to the bracket's riser face. Unlike the lidar mount's M2.5 count (a real Slamtec datasheet spec), this one **is** a picked default — the Arducam drawing didn't include an explicit hole-diameter callout in what was extracted, so M2.5 clearance was chosen to match the hardware already in this build (see Lidar Mount Hardware above), not re-measured from the drawing.

**Likely coverable by the same combined insert kit purchased for the lidar mount** — that kit only needed 4×M3 + 2×M2.5; a "~10-20pc" kit has headroom for this mount's 6×M3 + 4×M2.5 on top, meaning the two mounts probably don't need two separate kit purchases. Confirm total counts (10×M3 + 6×M2.5 across both mounts) against whatever specific kit size you actually buy.

| Component | Qty needed | Price (USD) | Notes |
|-----------|-----------|-----------|------|
| **M3 x 8mm socket head cap screws** (6 needed — bracket base flange, 2mm thick, so a shorter screw than the lidar mount's M3x12) | 6 | ~$8–12 | ⚠️ Price not confirmable via automated fetch — check at checkout; likely already covered by the assorted-length kit bought for the lidar mount |
| **M2.5 x 6mm socket head cap screws** (4 needed — Arducam board to the bracket's 2mm-thick riser face) | 4 | ~$8–12 | ⚠️ Price not confirmable via automated fetch — check at checkout; likely already covered by the assorted-length kit bought for the lidar mount |

**Estimated added cost:** ~$0 if buying assorted-length M3/M2.5 kits already covers this (likely, given the lidar mount already requires the same two screw sizes), otherwise ~$8–24 if buying separately.

---

## Why $300–350 isn't reachable right now

Doing the honest math, not the optimistic one (this table predates the 2026-08-11 camera swap — Kit price is real, camera price is now an open unknown rather than the old X10's confirmed $12.99, so treat the totals below as a floor, not a real total, until the Arducam board's price is checked at checkout):

| Scenario | Kit (fixed) | Camera | Pi 5 | RPLIDAR | Mount hardware (both) | **Total** |
|---|---|---|---|---|---|---|
| **Micro Center Marietta (8GB, recommended — confirm stock first)** | $179.00 | ⚠️ unconfirmed | $114.99 | $99.00 | $31–49 | **~$424–442 + camera** |
| Fallback: online only, no store trip (4GB) | $179.00 | ⚠️ unconfirmed | $110.00 | $99.00 | $31–49 | **~$419–437 + camera** |

Already **$74–142 above the $300–350 target** before the camera's own price is even added back in, even after real savings on the two components this pass was allowed to touch (RPLIDAR: −$20.98; Pi 5: −$60 to −$65 via the Micro Center trip). The reason isn't a research shortfall — it's that the Kit alone is already more than half the $300–350 target, leaving too little for a Raspberry Pi 5 *and* a genuine RPLIDAR A1M8 *and* both mounts' hardware *and* a camera, and the real current floor prices for the first three ($99–115 + $99 + $31–49 = ~$229–263 at minimum) don't fit in what's left no matter how much more searching happens — this was checked against multiple independent sources per component (three-plus resellers for the Pi, two independent official channels for the RPLIDAR), not a single quote taken at face value.

**Real levers left, if you want to keep pushing toward $300–350** (none of these were applied — they'd need your sign-off, since each is a bigger tradeoff than a price swap):
- **Drop the lidar variant entirely, ship camera-only.** The RPLIDAR ($99) and its mount hardware ($31–49) are the single biggest lever available — together they're most of the gap. This project's own `plan.md` already treats camera and lidar as two independent variants/policies, not a combined requirement, so a camera-only physical build is a real, already-supported option, not a new one.
- **A cheaper 2D lidar than the RPLIDAR A1M8** (e.g. Slamtec's own newer/cheaper models, or a different brand entirely) — not investigated this pass, since it would mean redesigning the physical mount (`lidar_mount_block_v1.stl` is dimensioned to the A1M8's exact datasheet geometry) and re-verifying the sim's lidar sensor config against different real specs — a real engineering change, not a pure price swap.
- **A used/refurbished Pi 5** — not investigated; secondhand marketplace listings carry real legitimacy/condition risk that "still legit" was meant to screen out, so this wasn't pursued without your explicit OK.

---

## Notes

- **LeKiwi Kit Stock:** The official Seeed Studio US store currently shows this as pre-order/limited availability. The Japanese distributor shows stock. Check [Seeed's official store](https://www.seeedstudio.com) for direct US availability, or consider the RobotShop/OpenELAB international options as fallback.

- **Camera (Arducam B0200, replaces the X10):** listed across Amazon, RobotShop, UCTronics, and Arducam's own store — genuinely available from multiple independent sellers, a good stock signal, but price could not be confirmed by automated fetch from any of them this session (see the camera-swap note at the top of this file). Check current price at whichever of the links above actually loads for you before ordering.

---

## Additional Items (Not in BoM, but needed for operation)

These are standard and typically already available:

- **Micro-USB or USB-C power adapter** for Raspberry Pi 5
- **microSD card** (≥32 GB recommended for LeRobot + OS)
- **USB hub or extension cable** (the Arducam camera and RPLIDAR both need USB ports; Pi 5 has 2×USB 3.0)
- **WiFi / Ethernet** for development and model deployment

---

## Assembly & Software References

- **Seeed LeKiwi Assembly:** [Seeed Wiki](https://wiki.seeedstudio.com/)
- **LeRobot Software Stack:** [LeRobot GitHub](https://github.com/huggingface/lerobot)
- **Arducam Camera Datasheet (B0201, same board family as the B0200 actually used — see camera-swap note above):** [uctronics.com PDF](https://www.uctronics.com/download/Amazon/B0201_IMX291_120_UVC_Camera_Datasheet.pdf) — UVC-compliant, no driver install needed on Linux (matches the Pi 5 + LeRobot stack the same as the X10 did)
- **RPLIDAR A1M8 Driver/SDK:** [Slamtec LiDAR Series Wiki](https://wiki.seeedstudio.com/slamtec/)

---

## Procurement Strategy

1. **Immediate (highest priority):** Check Micro Center Marietta's stock/pickup checker (or call) for the Raspberry Pi 5 8GB ($114.99) before ordering the 4GB fallback online — order the Arducam camera and RPLIDAR regardless (both ship, no store trip needed), but check the Arducam camera's real price at checkout first (unconfirmed by this session's tools).
   - Combined online shipping time (camera + RPLIDAR): ~7–10 days. Pi 5 same-day if Micro Center has stock, ~7–10 days if falling back to the online 4GB option.
   - Total cost to date: ~$213.99 + camera (Pi 8GB @ Micro Center + RPLIDAR, camera price still unconfirmed), or ~$209.00 + camera with the 4GB fallback — before mount hardware.

2. **Follow-up (monitor for stock):** LeKiwi Kit from official Seeed Studio or authorized distributor.
   - Lead time: 2–6 weeks depending on source.
   - Cost: ~$179.

3. **Once physical robot arrives:** Assemble per Seeed Wiki, install LeRobot + model policies from Phase 7.

---

*Last verified: August 11, 2026 (camera swap pass — X10 replaced with Arducam B0200, price unconfirmed; Kit/Pi 5/RPLIDAR prices carried over unchanged from the August 10, 2026 cost-reduction pass). Prices and stock status subject to change. Always confirm availability and current price before checkout.*
