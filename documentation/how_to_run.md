# How to Run - Lebanon Railway Network Map

## Quick Start

### Option 1: Direct Browser Opening (Simplest)
1. Save the HTML code as `lebanon-railway-map.html`
2. Double-click the file to open in your default browser
3. The map will load automatically

### Option 2: Drag and Drop
1. Save the HTML file
2. Open your web browser
3. Drag the HTML file into the browser window

---

## Detailed Setup Instructions

### Prerequisites
- **Web Browser**: Any modern browser (Chrome, Firefox, Safari, Edge)
- **Internet Connection**: Required for map tiles and Leaflet library
- **No Installation Required**: Everything runs in the browser

### Step-by-Step Guide

#### 1. Save the HTML File
```bash
# Create a new file named
lebanon-railway-map.html

# Copy the entire HTML code into this file
# Save the file
```

#### 2. File Structure (Optional)
If you want to organize your files:
```
lebanon-railway/
├── lebanon-railway-map.html
├── user-manual.md
└── how-to-run.md
```

#### 3. Opening the File

**Windows:**
- Right-click on `lebanon-railway-map.html`
- Select "Open with" → Choose your browser
- OR simply double-click the file

**macOS:**
- Double-click the HTML file
- OR right-click → "Open With" → Select browser

**Linux:**
- Double-click the file in file manager
- OR right-click → "Open With" → Web Browser
- OR use terminal: `xdg-open lebanon-railway-map.html`

---

## Running with a Local Server (Advanced)

### Why Use a Local Server?
- Better performance
- No CORS issues
- More professional development setup

### Option A: Python (Built-in)
```bash
# Python 3
cd /path/to/your/file
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000

# Open browser to: http://localhost:8000/lebanon-railway-map.html
```

### Option B: Node.js
```bash
# Install http-server globally
npm install -g http-server

# Run in your directory
cd /path/to/your/file
http-server -p 8080

# Open browser to: http://localhost:8080/lebanon-railway-map.html
```

### Option C: Live Server (VS Code)
1. Install "Live Server" extension in VS Code
2. Right-click on HTML file
3. Select "Open with Live Server"

### Option D: PHP
```bash
cd /path/to/your/file
php -S localhost:8000

# Open browser to: http://localhost:8000/lebanon-railway-map.html
```

---

## Hosting Online (Optional)

### GitHub Pages (Free)
1. Create a GitHub repository
2. Upload `lebanon-railway-map.html`
3. Rename to `index.html`
4. Enable GitHub Pages in Settings
5. Access at: `https://yourusername.github.io/repository-name`

### Netlify Drop (Free)
1. Visit [netlify.com/drop](https://app.netlify.com/drop)
2. Drag your HTML file to the browser
3. Get instant public URL

### CodePen
1. Create new pen at [codepen.io](https://codepen.io)
2. Paste HTML code in HTML section
3. Save and share the link

### Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd /path/to/your/file
vercel --prod
```

---

## Troubleshooting

### Map Not Loading?

**Check Internet Connection**
- The map requires internet for:
  - OpenStreetMap tiles
  - Leaflet.js library
  - Font Awesome icons

**Browser Console**
1. Press `F12` or right-click → "Inspect"
2. Go to "Console" tab
3. Look for error messages

**Common Fixes:**
```javascript
// If you see CORS errors, use a local server instead of file://
// If tiles don't load, check internet connection
// If JavaScript errors, ensure you copied the complete code
```

### Performance Issues?

**Browser Recommendations:**
- Chrome 90+ (Best performance)
- Firefox 88+ (Good performance)
- Safari 14+ (Good performance)
- Edge 90+ (Good performance)
- ❌ Internet Explorer (Not supported)

**Optimization Tips:**
1. Close unnecessary browser tabs
2. Disable browser extensions temporarily
3. Clear browser cache if needed
4. Use incognito/private mode for testing

---

## Development Setup

### For Modifications
1. Use a code editor (VS Code, Sublime, Atom)
2. Install helpful extensions:
   - HTML/CSS/JS syntax highlighting
   - Prettier for formatting
   - Live Server for auto-reload

### Project Structure
```
lebanon-railway-map.html
├── HTML Structure
├── CSS Styles (in <style> tag)
├── JavaScript (in <script> tag)
└── Leaflet.js Integration
```

### Making Changes
```javascript
// To modify railway lines, find this section:
const railwayLines = [
    {
        name: "Line Name",
        coordinates: [[lat, lon], ...],
        color: '#hexcolor',
        weight: lineThickness
    }
];

// To add stations, modify:
const stations = [
    { 
        name: "Station Name", 
        coords: [lat, lon], 
        type: "station-type" 
    }
];
```

---

## Browser Compatibility

### Fully Supported
| Browser | Minimum Version | Status |
|---------|----------------|---------|
| Chrome | 90+ | ✅ Excellent |
| Firefox | 88+ | ✅ Excellent |
| Safari | 14+ | ✅ Excellent |
| Edge | 90+ | ✅ Excellent |
| Opera | 76+ | ✅ Good |

### Mobile Browsers
| Browser | Platform | Status |
|---------|----------|---------|
| Chrome | Android | ✅ Full support |
| Safari | iOS | ✅ Full support |
| Firefox | Android | ✅ Full support |
| Samsung Internet | Android | ✅ Good |

### Not Supported
- Internet Explorer (any version)
- Browsers with JavaScript disabled
- Text-only browsers

---

## Quick Testing Checklist

- [ ] File saved as `.html`
- [ ] Internet connection active
- [ ] Modern browser installed
- [ ] JavaScript enabled
- [ ] Pop-up blocker not interfering
- [ ] Console shows no errors

---

## Additional Resources

### External Dependencies
The map automatically loads these from CDN:
- **Leaflet.js 1.9.4**: Map functionality
- **OpenStreetMap**: Map tiles
- **Font Awesome**: Icons (optional)

### No Installation Needed
Everything is loaded from CDN, so you only need:
1. The HTML file
2. Internet connection
3. Web browser

---

## Quick Commands Reference

```bash
# Quick run with Python
python -m http.server 8000

# Quick run with Node.js
npx http-server

# Quick run with PHP
php -S localhost:8000

# Open in default browser (Linux/Mac)
open lebanon-railway-map.html     # macOS
xdg-open lebanon-railway-map.html # Linux

# Open in default browser (Windows)
start lebanon-railway-map.html
```

That's it! The map should now be running in your browser. Enjoy exploring the Lebanon Railway Network!