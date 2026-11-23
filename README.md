# 🧠 Face Recognition System

A Flask + OpenCV web application for **face detection**, **eye detection**, **face recognition**, and **face comparison** — with a modern Tailwind CSS dashboard, SQLite history, and user login.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- 🔍 **Face Detection** — Haar Cascade based, three accuracy presets (low / medium / high)
- 👁️ **Eye Detection** — locates eyes within each detected face
- 🎯 **Face Recognition** — register identities and match uploaded faces (HOG + histogram descriptor)
- ⚖️ **Face Comparison** — ORB feature matching between two images
- 🗂️ **History & Dashboard** — every operation is stored in SQLite with stats
- 🔐 **Login System** — session-based auth (default: `admin` / `admin123`)
- 🎨 **Tailwind UI** — responsive glass-morphism dashboard
- 📥 **Download & preview** processed images

---

## 🗂️ Project Structure
