
---

> **This is a fork of [kabilankb/lekiwi_isaacsim](https://github.com/kabilankb/lekiwi_isaacsim)**, extended with an arm-free, camera-only LeKiwi variant built for training a navigation policy in Isaac Sim/Isaac Lab. See [`plan.md`](plan.md) for the full training-to-real-robot roadmap and [`isaac_sim/README_lekiwi_variants.md`](isaac_sim/README_lekiwi_variants.md) for how the USD variant was built and what it contains. The original repo's full arm+camera URDF/USD (with the SO-ARM101 attached) lives upstream at the link above, not duplicated here. A RPLIDAR A1M8 variant existed here previously; it's been removed while the camera variant's geometry is being brought in line with the real [Seeed LeKiwi Kit](https://www.seeedstudio.com/Lekiwi-Kit-p-6501.html) and will return later.

# 🦾 Lekiwi URDF for Isaac Sim

This repository provides the **Unified Robot Description Format (URDF)** models for the **Lekiwi robot**, optimized for use within **NVIDIA Isaac Sim**. Our goal is to enable realistic, high-fidelity simulations of the Lekiwi robot for robotics research, development, and testing.

![Image](https://github.com/user-attachments/assets/d5f6e9d0-4639-4994-b41d-35f78ffdd3ef)
---

## 📚 Table of Contents

* [About Lekiwi](#about-lekiwi)
* [Features](#features)
* [Getting Started](#getting-started)

  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Usage in Isaac Sim](#usage-in-isaac-sim)
  * [Viewing the URDF with ROS 2](#viewing-the-urdf-with-ros-2)
* [Current URDF Models](#current-urdf-models)
* [Upcoming Work](#upcoming-work)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

---

## 🤖 About Lekiwi

**Lekiwi** is a compact mobile manipulator designed for **indoor navigation** and **precise object manipulation**. This repository provides accurate URDF representations of Lekiwi for seamless integration into simulation environments like **Isaac Sim**.

---

## ✨ Features

* ✅ **Isaac Sim Compatible**: Designed and tested with NVIDIA Isaac Sim.
* 🧩 **Modular Design**: Easily modify or extend components.
* 🔧 **Accurate Kinematics & Dynamics**: Faithful representation of Lekiwi's real-world properties.
* 🎨 **Visual Meshes Included**: Realistic visuals using `.stl` or `.obj` meshes.

---

## 🚀 Getting Started

Follow these instructions to set up the URDF model locally for development and simulation.

### 🔧 Prerequisites

* **NVIDIA Isaac Sim** (See [Isaac Sim Docs](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html))
* **Python 3.x**
* **ROS 2** (`Humble`, `Iron`, or `Rolling` recommended)
* **urdf\_tutorial** package:

  ```bash
  sudo apt install ros-<ros2-distro>-urdf-tutorial
  ```

---

### 📦 Installation

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:kabilankb/lekiwi_isaacsim.git
   cd lekiwi_isaacsim
   ```

2. **Place URDF in Isaac Sim Assets:**
   ```

   **Direct Copy:**

   ```bash
   cp -r urdf meshes /path/to/isaac_sim_assets/lekiwi_robot/
   ```

---

### 🛠️ Usage in Isaac Sim

1. Launch Isaac Sim.
2. Go to `File -> New Stage`.
3. Import the URDF (`urdf/lekiwi_cam.urdf`):

   * `File -> Import -> URDF`
   * Enable **merge fixed joints** for better performance (optional)
4. Simulate and interact with the robot in the scene.

Or skip URDF import entirely and load `usd/lekiwi_camera.usd` directly — it's already a physics-ready Isaac Sim asset (real articulation, real camera mount), no import step needed. See `isaac_sim/README_lekiwi_variants.md`.

---

### 👁️ Viewing the URDF with ROS 2

1. **Source ROS 2:**

   ```bash
   source /opt/ros/<ros2-distro>/setup.bash
   ```

2. **Launch URDF in RViz:**

   ```bash
   ros2 launch urdf_tutorial display.launch.py model:=/absolute/path/to/lekiwi_isaacsim/urdf/lekiwi_cam.urdf
   ```
![Image](https://github.com/user-attachments/assets/ed895c3b-36ca-43e3-965a-78523df1fa8b)
---

## 📁 Current Models

* `urdf/lekiwi_cam.urdf` – Arm-free LeKiwi base URDF (Seeed X10 USB camera), references `urdf/meshes/`
* `usd/lekiwi_camera.usd` – The same variant as a self-contained, physics-ready Isaac Sim USD asset, derived from [LightwheelAI/leisaac](https://github.com/LightwheelAI/leisaac)'s real LeKiwi asset with the arm removed, wheel geometry cross-checked against the CAD-derived URDF, and the camera mount/model corrected to match the real Seeed Kit (see `isaac_sim/README_lekiwi_variants.md` for full provenance)
* `urdf/meshes/` – Visual and collision meshes (`.stl`)
* The original repo's full arm+camera `lekiwi.urdf`/`lekiwi.usd` (with the SO-ARM101 attached) is not included here — see [upstream](https://github.com/kabilankb/lekiwi_isaacsim) for that
* A RPLIDAR A1M8 variant (`lekiwi_lidar.urdf`/`.usd`) previously lived here; removed for now, will return once the camera variant is fully verified

---

## 🧪 Upcoming Work

* ⚙️ Refined collision meshes
* 📷 Sensor integration (LiDAR, cameras, IMU)
* 🔩 Actuator control enhancements (velocity/torque/friction)
* 🤖 ROS 2 control integration (`ros2_control`)
* 🌐 Task-specific Isaac Sim environments
* 📘 Expanded documentation and tutorials
* 🚀 Simulation performance optimization

---

## 🤝 Contributing

We welcome contributions from the community!

1. Fork the repo
2. Create a new branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add: Description of your feature"
   ```
4. Push and open a Pull Request.

Please follow coding standards and include relevant documentation. For major features, open an issue for discussion first.

---

## 📄 License

This project is licensed under the **[MIT License](LICENSE)**. You may modify and use it freely under the terms specified.

---

