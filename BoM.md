# LeKiwi Physical Robot — Bill of Materials

Final component list for Phase 9, with direct checkout links and verified stock status as of **August 2026**.

---

## Core Components

| Component | SKU / Model | Qty | Price (USD) | Stock Status | Link |
|-----------|-----------|-----|-----------|--------------|------|
| **Seeed LeKiwi Kit 12V** (mobile base, 3D printed parts, battery) | LeKiwi-12V | 1 | $179.00 | ⚠️ Pre-order / Limited | [Seeed Studio (JP)](https://jp.seeedstudio.com/mobile-base-c-2676.html) |
| **Raspberry Pi 5 8GB** (barebone board) | RPI5-8GB | 1 | $175.00–$200.00 | ✅ In stock (authorized resellers) | [Adafruit](https://www.adafruit.com/product/5813) |
| **Seeed X10 USB Camera 1080p** (front-facing RGB) | X10-USB | 1 | $12.99 | ✅ In stock | [Seeed Studio](https://www.seeedstudio.com/X10-USB-wired-camera-p-6506.html) |
| **RPLIDAR A1M8-R6 360° LiDAR** (360° 2D scanner, 12m range) | A1M8-R6 | 1 | $119.98 | ✅ In stock | [Walmart (LYK-Technology)](https://business.walmart.com/ip/RPLIDAR-A1M8-2D-360-Degree-12-Meters-Scanning-Radius-LIDAR-Sensor-Scanner-for-Obstacle-Avoidance-and-Navigation-of-Robots/14747563594) |

**Total Hardware Cost (excluding shipping, taxes):** ~$486.97–$511.97

---

## Lidar Mount Hardware (for the 3D-printed RPLIDAR A1M8 mount, `lidar_mount_block_v1.stl`)

The mount design uses **two different screw sizes**, not just M3 — worth knowing before
ordering, since it's easy to assume one size covers the whole assembly:

- **M3** for the 4 screws holding the mount block down to the robot's own top base
  plate (matches the plate's real 3.5mm hole grid).
- **M2.5**, not M3, for the 2 screws holding the RPLIDAR A1M8 itself to the block —
  that's Slamtec's own stock hole size on the real unit (confirmed in their datasheet),
  not a choice made for this build.

| Component | Qty needed | Price (USD) | Stock Status | Link |
|-----------|-----------|-----------|--------------|------|
| **M3 heat-set brass inserts** (100-pc pack, only need 4) | 1 pack | $10.99 | ✅ In stock | [Prusa3D](https://www.prusa3d.com/product/heat-set-inserts-m3-standard-100-pcs/) |
| **M2.5 heat-set brass inserts** (100-pc pack, only need 2) | 1 pack | $11.99 | ✅ In stock | [Prusa3D](https://www.prusa3d.com/product/threaded-inserts-m2-5-standard-100-pcs/) |
| **M3 x 12mm socket head cap screws** (for the 4 base-plate-to-block screws) | 4 | ~$8–12 for an assorted-length kit | ⚠️ Price not confirmable via automated fetch (Amazon loads pricing dynamically) — check at checkout | [Amazon: M3 assortment kit](https://www.amazon.com/Alloy-Socket-Assortment-Metric-Thread/dp/B01FH2365Q) |
| **M2.5 x 8mm socket head cap screws** (for the 2 lidar-to-block screws) | 2 | ~$8–12 for an assorted-length kit | ⚠️ Price not confirmable via automated fetch — check at checkout | [Amazon: M2.5 screw search](https://www.amazon.com/s?k=m2.5+socket+head+cap+screw) |
| **Heat-set insert installation tips + iron adapter** (M2/M3/M4/M5/M6/M8, fits TS100/TS101/Pinecil-style irons) | 1 kit | $32.40 | ✅ In stock | [kb-3d.com: CNCKitchen Heat Set Insert Tool Kit](https://kb-3d.com/store/tools-equipment/976-cnckitchen-heat-set-insert-tool-kit-ts100ts101-1688693301779.html) |

**Not a heat gun** — heat-set inserts need a **soldering iron** (controlled temperature,
firm even downward pressure to seat the insert square), not a heat gun; a heat gun can't
apply that directional force and tends to overheat/warp the surrounding plastic. If you
don't already own a soldering iron, the kit above assumes a TS100/TS101/Pinecil-style
iron (common, cheap, popular in the 3D-printing/maker world); if you have a different
iron (e.g. a Hakko or Weller), search for insert tips sized for that iron's tip mount
instead of buying the TS100-specific adapter above.

**Estimated added cost:** ~$55–75 (inserts + screws + tooling, assuming no soldering
iron is already owned beyond the tip/adapter kit).

---

## Notes

- **LeKiwi Kit Stock:** The official Seeed Studio US store currently shows this as pre-order/limited availability. The Japanese distributor shows stock. Check [Seeed's official store](https://www.seeedstudio.com) for direct US availability, or consider the RobotShop/OpenELAB international options as fallback.

- **Raspberry Pi 5 8GB:** Official MSRP is $95, but actual retail prices reflect high demand. Adafruit (authorized reseller) has consistent stock at $200; alternatives include SparkFun, CanaKit, and The Pi Hut. Micro Center occasionally discounts in-store to ~$65, but in-store only and subject to availability.

- **X10 Camera:** Direct from Seeed Studio at $12.99 is the cheapest clickable link; RobotShop lists $16.31 but may be out of stock.

- **RPLIDAR A1M8-R6:** Walmart B2B listing ($119.98) is the cheapest in-stock option with free US shipping. RobotShop ($103.12) is cheaper but currently "re-stocking soon." eBay and Amazon alternatives typically $130–$140 with free shipping.

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

1. **Immediate (highest priority):** Order Raspberry Pi 5 8GB from Adafruit, X10 camera from Seeed Studio, RPLIDAR from Walmart.
   - Combined shipping time: ~7–10 days.
   - Total cost to date: ~$308.

2. **Follow-up (monitor for stock):** LeKiwi Kit from official Seeed Studio or authorized distributor.
   - Lead time: 2–6 weeks depending on source.
   - Cost: ~$179.

3. **Once physical robot arrives:** Assemble per Seeed Wiki, install LeRobot + model policies from Phase 7.

---

*Last verified: August 9, 2026. Prices and stock status subject to change. Always confirm availability and current price before checkout.*
