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

*Last verified: August 8, 2026. Prices and stock status subject to change. Always confirm availability and current price before checkout.*
