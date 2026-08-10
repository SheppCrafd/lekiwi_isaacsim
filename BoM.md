# LeKiwi Physical Robot — Bill of Materials

Final component list for Phase 9, with direct checkout links and verified stock status as of **August 2026**.

**Cost-reduction pass (2026-08-10):** re-priced every component *except* the Seeed LeKiwi Kit and Seeed X10 camera against the least expensive still-legitimate option findable (no counterfeits/clone listings, no unverifiable marketplace sellers) — target was a **$300–350 total build cost**. Real savings were found (RPLIDAR down $20.98, Raspberry Pi down $65–90 depending on which variant), but **the honest total still lands at ~$436–456, not $300–350** — see "Why $300–350 isn't reachable right now" below before ordering anything on the assumption the target was hit.

---

## Core Components

| Component | SKU / Model | Qty | Price (USD) | Stock Status | Link |
|-----------|-----------|-----|-----------|--------------|------|
| **Seeed LeKiwi Kit 12V** (mobile base, 3D printed parts, battery) — *unchanged, excluded from this cost pass* | LeKiwi-12V | 1 | $179.00 | ⚠️ Pre-order / Limited | [Seeed Studio (JP)](https://jp.seeedstudio.com/mobile-base-c-2676.html) |
| **Seeed X10 USB Camera 1080p** (front-facing RGB) — *unchanged, excluded from this cost pass* | X10-USB | 1 | $12.99 | ✅ In stock | [Seeed Studio](https://www.seeedstudio.com/X10-USB-wired-camera-p-6506.html) |
| **Raspberry Pi 5 4GB** (barebone board) | RPI5-4GB | 1 | $110.00 | ✅ In stock (PiShop.us, official reseller) | [PiShop.us](https://www.pishop.us/product/raspberry-pi-5-4gb/) |
| **RPLIDAR A1M8-R6 360° LiDAR** (360° 2D scanner, 12m range) | A1M8-R6 | 1 | $99.00 | ✅ In stock | [Seeed Studio](https://www.seeedstudio.com/RPLiDAR-A1M8-R6-360-Degree-Laser-Scanner-Kit-12M-Range-p-4785.html) |

**Total Hardware Cost (excluding shipping, taxes, mount hardware below):** ~$400.99

### What changed and why

- **RPLIDAR: $119.98 → $99.00** (Walmart/LYK-Technology reseller → Seeed Studio direct). Real, confirmed floor price — cross-checked against Slamtec's own official AliExpress store, which also lists $99.00 for the same part, so this isn't a fluke listing, it's the genuine current price for a real (non-clone) A1M8. RobotShop's older $103.12 figure (cited in the previous pass) is no longer the cheapest verified option.
- **Raspberry Pi 5: $175–200 → $110.00**, by dropping from 8GB to **4GB**, not by finding a cheaper 8GB source. Every authorized 8GB reseller checked this pass (PiShop.us, CanaKit, SparkFun) is still clustered at **$175–180**, not the $95 official MSRP — this is a real, industry-wide condition (Raspberry Pi's own blog has publicly acknowledged 2025–2026 memory-cost-driven price increases on every 2GB+ SKU, calling it "painful but ultimately temporary"), not a search failure. 4GB is a genuine tradeoff, not free: this project's actual on-device workload is running a *trained* policy (inference only, no on-device training — see `plan.md` Phase 10), so 4GB is very likely sufficient, but hasn't been load-tested against the real LeRobot control stack. If you want to keep 8GB, see the Micro Center option below instead of paying $175 online.

### A real, location-dependent alternative for the Pi specifically

**Micro Center, in-store only:** 8GB board currently $114.99 (marked down from $125), confirmed on their own product page. That's *more RAM for barely more than the 4GB online price above* — genuinely worth it if you have a Micro Center within driving distance, since it beats every online 8GB price by $60+. Not usable if you don't (in-store purchase only, no reliable online ordering/shipping for this specific promo price, and it's a single physical chain with limited US locations). If accessible, this replaces the $110.00 line above with $114.99 for the 8GB variant instead of 4GB — a strictly better deal, not just a cheaper one.

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

## Why $300–350 isn't reachable right now

Doing the honest math, not the optimistic one:

| Scenario | Kit + Camera (fixed) | Pi 5 | RPLIDAR | Mount hardware | **Total** |
|---|---|---|---|---|---|
| Cheapest reliable online (4GB Pi) | $191.99 | $110.00 | $99.00 | $31–49 | **~$432–450** |
| Micro Center in-store (8GB Pi, if accessible) | $191.99 | $114.99 | $99.00 | $31–49 | **~$437–455** |

Both land **$82–155 above the $300–350 target**, even after real savings on the two components this pass was allowed to touch (RPLIDAR: −$20.98; Pi 5: −$60 to −$90 depending on variant/source). The reason isn't a research shortfall — it's that **$191.99 of the $300–350 target is already spent on the two fixed components alone**, leaving only $108–158 for a Raspberry Pi 5 *and* a genuine RPLIDAR A1M8 *and* mount hardware, and the real current floor prices for those three ($110 + $99 + $31 = $240 at minimum) don't fit in that remaining budget no matter how much more searching happens — this was checked against multiple independent sources per component (three-plus resellers for the Pi, two independent official channels for the RPLIDAR), not a single quote taken at face value.

**Real levers left, if you want to keep pushing toward $300–350** (none of these were applied — they'd need your sign-off, since each is a bigger tradeoff than a price swap):
- **Drop the lidar variant entirely, ship camera-only.** The RPLIDAR ($99) and its mount hardware ($31–49) are the single biggest lever available — together they're most of the gap. This project's own `plan.md` already treats camera and lidar as two independent variants/policies, not a combined requirement, so a camera-only physical build is a real, already-supported option, not a new one.
- **A cheaper 2D lidar than the RPLIDAR A1M8** (e.g. Slamtec's own newer/cheaper models, or a different brand entirely) — not investigated this pass, since it would mean redesigning the physical mount (`lidar_mount_block_v1.stl` is dimensioned to the A1M8's exact datasheet geometry) and re-verifying the sim's lidar sensor config against different real specs — a real engineering change, not a pure price swap.
- **Micro Center in-store, if there's one near you** — already the best lever actually applied above; ask if you want the exact current in-store 4GB price checked too (only the 8GB in-store price was confirmed this pass).
- **A used/refurbished Pi 5** — not investigated; secondhand marketplace listings carry real legitimacy/condition risk that "still legit" was meant to screen out, so this wasn't pursued without your explicit OK.

---

## Notes

- **LeKiwi Kit Stock:** The official Seeed Studio US store currently shows this as pre-order/limited availability. The Japanese distributor shows stock. Check [Seeed's official store](https://www.seeedstudio.com) for direct US availability, or consider the RobotShop/OpenELAB international options as fallback.

- **X10 Camera:** Direct from Seeed Studio at $12.99 is the cheapest clickable link; RobotShop lists $16.31 but may be out of stock.

---

## Additional Items (Not in BoM, but needed for operation)

These are standard and typically already available:

- **Micro-USB or USB-C power adapter** for Raspberry Pi 5
- **microSD card** (≥32 GB recommended for LeRobot + OS)
- **USB hub or extension cable** (X10 camera and RPLIDAR both need USB ports; Pi 5 has 2×USB 3.0)
- **WiFi / Ethernet** for development and model deployment

---

## Assembly & Software References

- **Seeed LeKiwi Assembly:** [Seeed Wiki](https://wiki.seeedstudio.com/)
- **LeRobot Software Stack:** [LeRobot GitHub](https://github.com/huggingface/lerobot)
- **X10 Camera Driver/Setup:** [Seeed X10 Documentation](https://files.seeedstudio.com/Bazaar/product_pdf/114090066.pdf)
- **RPLIDAR A1M8 Driver/SDK:** [Slamtec LiDAR Series Wiki](https://wiki.seeedstudio.com/slamtec/)

---

## Procurement Strategy

1. **Immediate (highest priority):** Order Raspberry Pi 5 4GB from PiShop.us (or check Micro Center in-store for 8GB first), X10 camera from Seeed Studio, RPLIDAR from Seeed Studio.
   - Combined shipping time: ~7–10 days.
   - Total cost to date: ~$222–227 (Pi + camera + RPLIDAR only, before mount hardware).

2. **Follow-up (monitor for stock):** LeKiwi Kit from official Seeed Studio or authorized distributor.
   - Lead time: 2–6 weeks depending on source.
   - Cost: ~$179.

3. **Once physical robot arrives:** Assemble per Seeed Wiki, install LeRobot + model policies from Phase 7.

---

*Last verified: August 10, 2026 (cost-reduction pass — Pi 5 and RPLIDAR re-priced, mount hardware re-estimated; Kit and X10 camera prices carried over unchanged from August 9, 2026). Prices and stock status subject to change. Always confirm availability and current price before checkout.*
