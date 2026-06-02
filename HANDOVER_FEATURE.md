# Cell Tower Handover Visualization Feature

## Overview
This feature allows you to visualize real cell tower data transferring between two geographic areas, simulating network handover scenarios.

## How to Use

### 1. **Access the Map View**
   - Click on the "MAP" tab in the navigation bar
   - The Global Network Traffic Map will load

### 2. **Activate Handover Mode**
   - Click the orange **"HANDOVER"** button in the map overlay
   - A control panel will appear on the right side

### 3. **Select Area 1 (Source)**
   - Enter a location name in the "Source Area" field (e.g., "Mumbai, India", "New York", "London")
   - Click **"SELECT AREA 1"** button
   - The system will geocode the location and show:
     - ✓ Status indicator when Area 1 is set
     - Real cell towers from that location

### 4. **Select Area 2 (Target)**
   - Enter a different location in the "Target Area" field (e.g., "Bangalore, India", "Los Angeles")
   - Click **"SELECT AREA 2"** button
   - Status updates to show both areas are selected

### 5. **Visualize the Handover**
   - Click **"VISUALIZE HANDOVER"** button
   - The map will:
     - **Fit both areas** in the viewport
     - **Display towers** from both areas with color coding:
       - **Cyan glowing borders**: Source area towers (Area 1)
       - **Orange glowing borders**: Target area towers (Area 2)
     - **Show connection lines** between towers
     - **Animate data flow** with orange particles flowing from Area 1 to Area 2

## What You'll See

### Tower Markers
- **Source Towers (Area 1)**
  - Marked with blue "T" icon
  - Surrounded by cyan aura
  - Shows operator, radio type, and estimated load

- **Target Towers (Area 2)**
  - Marked with orange "T" icon
  - Surrounded by orange aura
  - Shows operator, radio type, and estimated load

### Connection Visualization
- **Handover Line**: Dashed orange line connecting towers
- **Data Flow Animation**: Moving particles showing data transfer direction
- **Area Markers**: Numbered circles (1 and 2) marking area centers

### Status Panel
Shows:
- ✓ Area 1 and Area 2 completion status
- Total tower count in each area
- Estimated network load (Mbps) in each area
- Active handover status

## Real Data Source
The feature uses **real OpenCelliD tower inventory** which provides:
- Actual cell tower locations (lat/lng)
- Radio technology (2G/3G/4G/5G)
- Operator information (Jio, Airtel, Vi, BSNL, MTNL)
- Estimated network utilization (inferred from tower metadata)

## Example Scenarios

### Scenario 1: City-to-City Handover
- Area 1: **Mumbai** (Jio towers - blue)
- Area 2: **Bangalore** (Vi towers - pink)
- Shows handover from Western to Southern India

### Scenario 2: Cross-Country Network
- Area 1: **San Francisco** (Verizon/AT&T towers)
- Area 2: **New York** (Sprint/T-Mobile towers)
- Shows cross-country mobile user handover

### Scenario 3: Regional Coverage
- Area 1: **London, UK** (O2/Vodafone towers)
- Area 2: **Dublin, Ireland** (Three/Eir towers)
- Shows Irish roaming handover

## Features Displayed

✓ **Real Tower Inventory** - From OpenCelliD database
✓ **Network Operator Color Coding** - Distinguish between carriers
✓ **Traffic Load Estimation** - Shows inferred network capacity
✓ **Animated Handover Flow** - Visual representation of data transfer
✓ **Multi-Area Comparison** - Side-by-side tower analysis
✓ **Interactive Popups** - Click towers for detailed information
✓ **Dynamic Status Updates** - Real-time visualization status

## Technical Details

### API Endpoints Used
- `/api/towers?lat={lat}&lng={lng}` - Fetches real OpenCelliD tower data

### Geocoding Services
- Primary: OpenStreetMap Nominatim
- Fallback: ArcGIS World Geocoder

### Supported Operators (Color Coded)
- 🔵 **Jio** (MCC 404/405) - Cyan
- 🟠 **Airtel** (MCC 404/405) - Orange
- 🔴 **Vi/Vodafone** (MCC 404/405) - Magenta
- 🟢 **BSNL** (MCC 404) - Green
- 🟡 **MTNL** (MCC 404) - Yellow

## Tips & Tricks

1. **Get Accurate Results**: Use full city names (e.g., "Mumbai, Maharashtra, India")
2. **Zoom In**: After visualization, scroll to zoom in on specific towers
3. **Click Towers**: Click any tower marker to see detailed information
4. **Multiple Handovers**: Clear and select new areas to visualize different routes
5. **Network Analysis**: Compare tower density and load between areas

## Troubleshooting

### "Location Not Found"
- Try using a more specific location name
- Use format: "City, Country" (e.g., "Paris, France")

### No Towers Appear
- The area may have limited tower coverage in OpenCelliD database
- Try a larger city or densely populated area
- Check internet connection to OpenCelliD API

### Map Not Updating
- Ensure the Flask backend is running (`python app.py`)
- Check browser console for errors (F12 → Console tab)

## Future Enhancements

Planned features:
- Live handover metrics (signal strength, latency)
- 3D tower visualization
- Historical handover statistics
- Network congestion heatmaps
- Signal strength prediction
- Roaming cost analysis
