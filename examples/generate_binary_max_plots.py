import matplotlib.pyplot as plt
import numpy as np

# Set a professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# --- Data Extraction from User's text ---

# Data for Chart 1: Maximum compositions for V-X binary alloys
max_fractions = {
    'Cr': 0.4971,
    'Ti': 0.6972,
    'W': 0.9500,
    'Zr': 0.0493
}

# Data for Chart 2: Gas production and Dose rate
materials = ['V', 'Cr', 'Ti', 'W', 'Zr']

# Gas production data (He and H in appm)
gas_production = {
    'V': {'He': 244.52, 'H': 635.76},
    'Cr': {'He': 477.58, 'H': 1770.79},
    'Ti': {'He': 501.38, 'H': 1445.11},
    'W': {'He': 9.56, 'H': 47.57},
    'Zr': {'He': 164.49, 'H': 514.18}
}

# Dose rate data (Sv/h) at different cooling times (in days)
dose_rates = {
    'V': {'30 days': 0.013919599152391583, '365 days': 2.4011751643367956e-05, '1825 days': 3.029438835700662e-08, '36500 days': 4.075920006956689e-09},
    'Cr': {'30 days': 0.8636562634934948, '365 days': 0.00024009839328002942, '1825 days': 1.388597687968871e-06, '36500 days': 5.0686617338504686e-11},
    'Ti': {'30 days': 6.459089435152961, '365 days': 0.4014972333205736, '1825 days': 0.00023419569056964365, '36500 days': 3.141376917685182e-05},
    'W': {'30 days': 9.434409465139566, '365 days': 0.3866656570602694, '1825 days': 5.5820319441895954e-05, '36500 days': 4.00860744093665e-11},
    'Zr': {'30 days': 20268.468062418488, '365 days': 0.6047189151576632, '1825 days': 4.624079130919152e-06, '36500 days': 8.400192950675306e-09}
}

# --- Plotting Chart 1: Maximum Compositions ---
fig1, ax1 = plt.subplots(figsize=(8, 6))

elements = list(max_fractions.keys())
fractions = list(max_fractions.values())

bars = ax1.bar(elements, fractions, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], edgecolor='black', zorder=2)
ax1.set_xlabel('Alloying Element', fontsize=12, fontweight='bold')
ax1.set_ylabel('Maximum Alloying Fraction', fontsize=12, fontweight='bold')
ax1.set_title('Maximum Alloying Fraction in V-X Binary Alloys', fontsize=14, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=10)

# Add data labels to bars (rounded to 3 decimals)
for bar, value in zip(bars, fractions):
    height = bar.get_height()
    ax1.annotate(
        f"{value:.3f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=10,
        fontweight='bold',
        color='black'
    )

# Add a horizontal line to represent a baseline or limit if needed, e.g., V: 0.9500
# ax1.axhline(y=0.95, color='gray', linestyle='--', label='V fraction limit')
# ax1.legend()
plt.tight_layout()
plt.savefig('max_compositions_chart.png', dpi=300)
plt.show()

# --- Plotting Chart 2: Gas Production and Dose Rate ---
fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot A: Gas Production
bar_width = 0.35
x = np.arange(len(materials))
he_appm = [gas_production[m]['He'] for m in materials]
h_appm = [gas_production[m]['H'] for m in materials]

bars_he = ax2.bar(x - bar_width/2, he_appm, bar_width, label='He appm', color='#1f77b4', edgecolor='black', zorder=2)
bars_h = ax2.bar(x + bar_width/2, h_appm, bar_width, label='H appm', color='#ff7f0e', edgecolor='black', zorder=2)

ax2.set_xlabel('Material', fontsize=12, fontweight='bold')
ax2.set_ylabel('Gas Production (appm)', fontsize=12, fontweight='bold')
ax2.set_title('Gas Production in V-X Binary Alloys', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(materials)
ax2.legend(fontsize=10)
ax2.tick_params(axis='both', which='major', labelsize=10)

# Add data labels to He bars (rounded to 1 decimal)
for bar, value in zip(bars_he, he_appm):
    height = bar.get_height()
    ax2.annotate(
        f"{value:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=9,
        color='black'
    )

# Add data labels to H bars (rounded to 1 decimal)
for bar, value in zip(bars_h, h_appm):
    height = bar.get_height()
    ax2.annotate(
        f"{value:.1f}",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 5),
        textcoords="offset points",
        ha='center',
        va='bottom',
        fontsize=9,
        color='black'
    )

# Subplot B: Dose Rate
cooling_times = list(list(dose_rates.values())[0].keys())
num_times = len(cooling_times)
bar_width = 0.2
x = np.arange(len(materials))

dose_data = {
    time: [dose_rates[m][time] for m in materials] for time in cooling_times
}

for i, time in enumerate(cooling_times):
    ax3.bar(x + i*bar_width - (num_times-1)*bar_width/2, dose_data[time], bar_width, label=time, zorder=2)

ax3.set_xlabel('Material', fontsize=12, fontweight='bold')
ax3.set_ylabel('Dose Rate ($Sv/h$)', fontsize=12, fontweight='bold')
ax3.set_title('Dose Rate at Various Cooling Times', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(materials)
ax3.set_yscale('log')  # Use a logarithmic scale for the y-axis
ax3.legend(fontsize=10, title="Cooling Times")
ax3.tick_params(axis='both', which='major', labelsize=10)
ax3.set_ylim(1e-12, 1e5) # Set explicit y-axis limits to avoid plotting errors with log scale

plt.tight_layout()
plt.savefig('gas_production_dose_rate_chart.png', dpi=300)
plt.show()

