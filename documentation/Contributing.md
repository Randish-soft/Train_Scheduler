# Contributing to Lebanon Railway Network Map

Thank you for your interest in contributing to the Lebanon Railway Network Map project! This visualization aims to showcase a comprehensive railway system for Lebanon, and we welcome contributions that improve accuracy, functionality, and user experience.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Guidelines](#development-guidelines)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)
- [Adding New Features](#adding-new-features)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inspiring community for all. Contributors are expected to:
- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive criticism
- Show empathy towards other community members

### Unacceptable Behavior
- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Any conduct which would be considered inappropriate in a professional setting

---

## How Can I Contribute?

### 1. 🗺️ Map Data Improvements
- **Station Locations**: Verify and correct GPS coordinates
- **Route Accuracy**: Improve railway line paths based on topography
- **Missing Stations**: Add local or planned stations
- **Infrastructure**: Add missing tunnels, bridges, or viaducts

### 2. 🚀 Feature Enhancements
- **Interactivity**: Add new interactive elements
- **Visualizations**: Implement elevation profiles, speed charts
- **Information**: Expand popup content with more details
- **Accessibility**: Improve keyboard navigation and screen reader support

### 3. 🎨 Design Improvements
- **UI/UX**: Enhance visual design and user interface
- **Mobile**: Optimize for mobile devices
- **Themes**: Add dark mode or alternative themes
- **Icons**: Create custom station or train icons

### 4. 📚 Documentation
- **Translations**: Translate documentation to Arabic or French
- **Tutorials**: Create video tutorials or guides
- **Examples**: Add use case examples
- **API Documentation**: Document code functions

### 5. 🐛 Bug Fixes
- Fix rendering issues
- Resolve browser compatibility problems
- Correct data inconsistencies
- Improve performance

---

## Getting Started

### 1. Fork the Repository
```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/lebanon-railway-map.git
cd lebanon-railway-map

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/lebanon-railway-map.git
```

### 2. Create a Branch
```bash
# Create a new branch for your feature
git checkout -b feature/station-updates

# Or for bug fixes
git checkout -b fix/mobile-rendering
```

### 3. Set Up Development Environment
```bash
# No build process required - just need a local server
python -m http.server 8000
# Visit http://localhost:8000
```

---

## Development Guidelines

### Project Structure
```
lebanon-railway-map/
├── index.html              # Main map file
├── README.md              # Project overview
├── CONTRIBUTING.md        # This file
├── docs/
│   ├── user-manual.md     # User documentation
│   └── how-to-run.md      # Setup instructions
└── data/
    ├── stations.json      # Station data (if separated)
    └── routes.json        # Route data (if separated)
```

### Code Organization
```javascript
// Structure within index.html

// 1. Configuration
const CONFIG = {
    center: [33.8547, 35.8623],
    zoom: 9,
    maxZoom: 18
};

// 2. Data
const railwayLines = [...];
const stations = [...];
const tunnels = [...];
const bridges = [...];

// 3. Map initialization
const map = L.map('map', CONFIG);

// 4. Layer creation functions
function createStationMarker(station) {...}
function createRailwayLine(line) {...}

// 5. Event handlers
function handleToggle(layerName) {...}

// 6. Initialization
initializeMap();
```

### Data Format Standards

#### Railway Lines
```javascript
{
    name: "Line Name",              // Clear, descriptive name
    coordinates: [                  // Array of [lat, lon] pairs
        [33.8886, 35.4955],
        [33.9200, 35.5300]
    ],
    color: '#ff0000',              // Hex color code
    weight: 6,                     // Line thickness (4-8)
    opacity: 0.8,                  // Transparency (0.6-1.0)
    metadata: {                    // Optional additional data
        maxSpeed: 200,
        electrified: true,
        doubleTrack: true
    }
}
```

#### Stations
```javascript
{
    name: "Station Name",          // Official station name
    coords: [33.8886, 35.4955],    // [latitude, longitude]
    type: "major",                 // mega|major|regional|local|freight
    platforms: 6,                  // Number of platforms
    passengers: "45k/day",         // Daily passenger estimate
    facilities: [                  // Optional facilities
        "parking", "airport-link", "bus-terminal"
    ],
    elevation: 945                 // Meters above sea level
}
```

#### Tunnels
```javascript
{
    name: "Tunnel Name",
    coords: [                      // Start and end coordinates
        [33.9811, 35.6178],
        [34.0200, 35.6400]
    ],
    length: "12.3 km",            // Length with unit
    type: "Twin-bore",            // Twin-bore|Single-bore
    depth: 180,                   // Maximum depth in meters
    year: 2027                    // Expected completion
}
```

---

## Submitting Changes

### 1. Commit Messages
Follow the conventional commits format:
```bash
# Format: <type>(<scope>): <subject>

feat(stations): add Byblos suburban stations
fix(map): correct Tripoli station coordinates
docs(readme): update installation instructions
style(ui): improve mobile responsiveness
refactor(data): separate routes into JSON file
test(routes): add validation for coordinates
chore(deps): update Leaflet to 1.9.4
```

### 2. Pull Request Process

#### Before Submitting
- [ ] Test in multiple browsers (Chrome, Firefox, Safari)
- [ ] Verify mobile responsiveness
- [ ] Run code through validator
- [ ] Update documentation if needed
- [ ] Add comments for complex code
- [ ] Ensure no console errors

#### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tested on Chrome
- [ ] Tested on Firefox
- [ ] Tested on mobile
- [ ] No console errors

## Screenshots
(If applicable)

## Related Issues
Fixes #123
```

### 3. Review Process
1. Submit PR with clear description
2. Respond to review comments
3. Make requested changes
4. Squash commits if requested
5. Celebrate when merged! 🎉

---

## Style Guidelines

### JavaScript
```javascript
// Use ES6+ features
const stationData = stations.map(station => ({
    ...station,
    id: generateId()
}));

// Clear variable names
const beirutCentralStation = findStation('Beirut Central');

// Document complex functions
/**
 * Calculate optimal route between two stations
 * @param {Object} origin - Starting station
 * @param {Object} destination - End station
 * @returns {Array} Array of coordinates
 */
function calculateRoute(origin, destination) {
    // Implementation
}

// Use consistent formatting
if (condition) {
    doSomething();
} else {
    doSomethingElse();
}
```

### CSS
```css
/* Component-based naming */
.station-marker {
    /* Styles */
}

.station-marker--major {
    /* Major station variant */
}

.station-marker__icon {
    /* Icon within marker */
}

/* Consistent spacing */
.control-panel {
    padding: 15px;
    margin-bottom: 10px;
}

/* Mobile-first approach */
.map-container {
    height: 100vh;
}

@media (max-width: 768px) {
    .map-container {
        height: 60vh;
    }
}
```

### HTML
```html
<!-- Semantic HTML -->
<section class="controls">
    <h2>Layer Controls</h2>
    <button id="toggle-stations" class="control-btn">
        Stations
    </button>
</section>

<!-- Accessibility -->
<button aria-label="Toggle station visibility" role="switch">
    <span class="sr-only">Toggle stations</span>
</button>
```

---

## Adding New Features

### Feature Checklist
1. **Plan the Feature**
   - Define requirements
   - Consider edge cases
   - Design UI/UX

2. **Implement**
   - Write clean, documented code
   - Follow existing patterns
   - Add error handling

3. **Test**
   - Manual testing
   - Cross-browser testing
   - Performance testing

4. **Document**
   - Add inline comments
   - Update user manual
   - Add examples

### Example: Adding a New Station Type
```javascript
// 1. Define the new type
const STATION_TYPES = {
    mega: { size: 30, icon: '🚄' },
    major: { size: 25, icon: '🚅' },
    regional: { size: 20, icon: '🚉' },
    tourist: { size: 22, icon: '🎫' }, // New type
    local: { size: 18, icon: '🚏' }
};

// 2. Add station data
{
    name: "Beiteddine Palace",
    coords: [33.6950, 35.5806],
    type: "tourist",
    platforms: 2,
    passengers: "10k/day",
    attractions: ["Palace", "Museum", "Gardens"]
}

// 3. Update popup content
if (station.type === 'tourist' && station.attractions) {
    popupContent += `
        <div class="attractions">
            <strong>Nearby:</strong>
            ${station.attractions.join(', ')}
        </div>
    `;
}
```

---

## Reporting Issues

### Before Reporting
1. Check existing issues
2. Verify it's reproducible
3. Gather system information

### Issue Template
```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Go to...
2. Click on...
3. See error...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- Browser: Chrome 96
- OS: Windows 10
- Screen size: 1920x1080

## Screenshots
(If applicable)

## Console Errors
```
Any error messages
```
```

### Feature Requests
```markdown
## Feature Description
What feature would you like to see?

## Use Case
Why is this feature needed?

## Proposed Solution
How might this work?

## Alternatives Considered
Other options explored

## Additional Context
Any mockups, examples, or references
```

---

## Recognition

### Contributors
All contributors will be recognized in:
- README.md contributors section
- Code comments for significant contributions
- Project documentation

### Types of Recognition
- 🌟 Code Contributors
- 📚 Documentation Writers
- 🐛 Bug Hunters
- 🎨 Design Contributors
- 🌍 Translators
- 💡 Idea Contributors

---

## Questions?

If you have questions about contributing:
1. Check existing documentation
2. Look through closed issues
3. Open a new discussion issue
4. Tag it with `question`

We're here to help you contribute successfully!

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (specify your license here).

Thank you for helping improve the Lebanon Railway Network Map! 🚄🇱🇧