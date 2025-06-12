# 🌍 Semiconductor Trade Analysis Dashboard

A comprehensive Streamlit dashboard for analyzing US semiconductor trade data using official Census Bureau APIs and classification systems.

## 📊 Features

- **HS Code Analysis**: Product-based classification (HS 8541+8542 semiconductor products)
- **NAICS Code Analysis**: Industry-based classification (NAICS 334413 semiconductor manufacturing)
- **Comparative Analysis**: Direct comparison between HS and NAICS data
- **Real-time Data**: Live API calls to Census Bureau trade databases
- **Interactive Visualizations**: Charts, trends, and geographic breakdowns
- **Data Export**: Download results as CSV files

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Census Bureau API key (free at [api.census.gov](https://api.census.gov/data/key_signup.html))

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/semiconductor-trade-dashboard.git
   cd semiconductor-trade-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key:**
   ```bash
   export CENSUS_API_KEY="your_api_key_here"
   ```

4. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub repository
   - Set main file path: `app.py`

3. **Configure secrets:**
   - In Streamlit Cloud, go to App settings → Secrets
   - Add your API key:
     ```toml
     CENSUS_API_KEY = "your_api_key_here"
     ```

## 📋 Data Sources

### Primary APIs
- **Census Bureau International Trade API**
  - HS Code exports: `api.census.gov/data/timeseries/intltrade/exports/hs`
  - HS Code imports: `api.census.gov/data/timeseries/intltrade/imports/hs`
  - NAICS exports: `api.census.gov/data/timeseries/intltrade/exports/naics`
  - NAICS imports: `api.census.gov/data/timeseries/intltrade/imports/naics`

### Classification Systems
- **HS 8541**: Diodes, transistors, and similar semiconductor devices
- **HS 8542**: Electronic integrated circuits
- **NAICS 334413**: Semiconductor and Related Device Manufacturing

## 🔧 Configuration

### Environment Variables
- `CENSUS_API_KEY`: Your Census Bureau API key (required)

### Streamlit Secrets (for cloud deployment)
Create `.streamlit/secrets.toml`:
```toml
CENSUS_API_KEY = "your_api_key_here"
```

## 📊 Usage

### Data Source Options
1. **HS Codes**: Analyze semiconductor products crossing borders
2. **NAICS Codes**: Analyze semiconductor manufacturing activity
3. **Both**: Compare product vs manufacturing data

### Analysis Features
- **Year Range Selection**: Choose from 2013-2024
- **Trade Type**: Exports, imports, or both
- **Country Analysis**: Top trading partners
- **Trend Visualization**: Monthly time series charts
- **Trade Balance**: Calculate surplus/deficit
- **Data Validation**: Cross-reference with official totals

## 🌍 Country Coverage

The dashboard includes comprehensive country code mapping covering:
- 🇺🇸 North America (US, Canada, Mexico)
- 🇪🇺 Europe (All EU countries + UK, Switzerland, etc.)
- 🇨🇳 Asia & Middle East (China, Japan, South Korea, etc.)
- 🇧🇷 South America (Brazil, Argentina, Chile, etc.)
- 🇦🇺 Oceania (Australia, New Zealand, Pacific Islands)
- 🇿🇦 Africa (All African countries)

## 📈 Key Insights

### Semiconductor Trade Patterns
- US is typically a net importer of semiconductor products
- Major export destinations: Mexico, China, Hong Kong
- Major import sources: Malaysia, China, Taiwan
- Manufacturing vs product trade shows different patterns

### Data Validation
- HS and NAICS data correlation analysis
- Official totals cross-verification
- Identification of data gaps or anomalies

## 🛠️ Technical Details

### API Rate Limiting
- Built-in request delays to respect Census API limits
- Progress bars for long-running queries
- Error handling and retry logic

### Data Processing
- Automatic country code mapping
- Date standardization and validation
- Trade value aggregation and analysis
- Export-ready CSV formatting

### Caching
- Streamlit caching for improved performance
- 1-hour TTL for API responses
- Efficient data reuse across sessions

## 📝 File Structure

```
semiconductor-trade-dashboard/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore patterns
└── .streamlit/
    └── secrets.toml     # Local secrets (not committed)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### API Key Issues
- Get free key: [Census Bureau API signup](https://api.census.gov/data/key_signup.html)
- Check key validity by testing API calls
- Ensure proper environment variable setup

### Data Issues
- Verify year ranges (2013-2024 supported)
- Check trade type selections
- Review country code mappings

### Deployment Issues
- Ensure all dependencies in requirements.txt
- Verify secrets configuration in Streamlit Cloud
- Check GitHub repository permissions

## 🔗 Links

- **Live Dashboard**: [Your deployed app URL]
- **Census Bureau API**: [api.census.gov](https://api.census.gov)
- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Data Sources**: [USITC DataWeb](https://dataweb.usitc.gov)

## 📊 Sample Outputs

The dashboard provides:
- Interactive line charts showing trade trends
- Country-by-country trade breakdowns
- Trade balance calculations
- CSV exports for further analysis
- Data validation against official sources

---

**Built with:** 🐍 Python • 📊 Streamlit • 🌐 Census Bureau APIs • 📈 Pandas • 📉 Matplotlib