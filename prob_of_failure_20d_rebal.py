import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)

def futures_portfolio_simulation_with_rebalancing(
    initial_capital=100.0,
    target_leverage=1.6,
    mu_annual=0.10,
    sigma_annual=0.10,
    years=5,
    trading_days=252,
    rebalancing_frequency=20,
    init_margin_rate=0.137,
    maint_margin_rate=0.122,
    funding_rate_annual=0.068,
    rebalancing_cost_bp=2,
    t_df=4,
    simulations=10000,
    export_sample_size=1000,
    random_seed=42
):
    """
    Monte Carlo simulation with 20-day rebalancing to target leverage.
    Returns both plotting data and detailed path information for Excel export.
    """
    
    # Calculate derived parameters
    mu_daily = mu_annual / trading_days
    sigma_daily = sigma_annual / np.sqrt(trading_days)
    funding_daily = funding_rate_annual / trading_days
    total_days = trading_days * years
    rebalancing_cost_rate = rebalancing_cost_bp / 10000
    
    print(f"Simulation Configuration - Target Leverage {target_leverage}x:")
    print(f"- Rebalancing frequency: Every {rebalancing_frequency} trading days")
    print(f"- Total rebalancing events: {total_days // rebalancing_frequency}")
    print(f"- Rebalancing cost: {rebalancing_cost_bp}bp per turnover")
    
    # Results storage
    results = {
        'failures': 0,
        'margin_call_paths': 0,
        'total_margin_calls': 0,
        'final_equities': [],
        'max_drawdowns': [],
        'total_interests': [],
        'total_rebalancing_costs': [],
        'sample_paths': [],  # For plotting
        'detailed_paths': {},  # For Excel export
        'path_summaries': []
    }
    
    # Select paths for detailed tracking and plotting
    export_indices = set(np.random.choice(simulations, export_sample_size, replace=False))
    plot_indices = set(np.random.choice(simulations, 50, replace=False))
    
    # Run simulations
    for sim in range(simulations):
        # Initialize simulation variables
        equity = initial_capital
        debt = 0.0
        peak_equity = initial_capital
        max_drawdown = 0.0
        margin_calls_count = 0
        total_interest = 0.0
        total_rebalancing_cost = 0.0
        failed = False
        failure_day = None
        
        # Initial notional exposure
        notional = target_leverage * equity
        
        # Track paths for plotting
        if sim in plot_indices:
            equity_path = [equity]
            days_path = [0]
        
        # Detailed recording setup
        record_details = sim in export_indices
        if record_details:
            daily_records = []
            rebalancing_events = []
        
        # Daily simulation loop
        for day in range(1, total_days + 1):
            # Check for rebalancing day
            is_rebalancing_day = (day % rebalancing_frequency == 0)
            
            # Generate daily return with t-distribution
            z = stats.t.rvs(df=t_df)
            z *= np.sqrt((t_df - 2) / t_df)  # Normalize variance to 1
            daily_return = mu_daily + sigma_daily * z
            
            # Calculate P&L
            daily_pnl = notional * daily_return
            equity_before = equity
            equity += daily_pnl
            
            # Check for immediate failure
            if equity <= 0:
                results['failures'] += 1
                failed = True
                failure_day = day
                break
            
            # Rebalancing logic
            rebalancing_cost = 0
            if is_rebalancing_day and not failed:
                new_target_notional = target_leverage * equity
                notional_change = abs(new_target_notional - notional)
                
                # Apply rebalancing cost
                rebalancing_cost = notional_change * rebalancing_cost_rate
                equity -= rebalancing_cost
                total_rebalancing_cost += rebalancing_cost
                
                # Update notional
                old_notional = notional
                notional = new_target_notional
                
                # Record rebalancing event
                if record_details:
                    rebalancing_events.append({
                        'day': day,
                        'old_notional': old_notional,
                        'new_notional': notional,
                        'notional_change': notional_change,
                        'rebalancing_cost': rebalancing_cost,
                        'equity_after_rebalancing': equity
                    })
                
                # Check for failure after rebalancing
                if equity <= 0:
                    results['failures'] += 1
                    failed = True
                    failure_day = day
                    break
            
            # Calculate margin requirements
            initial_required = init_margin_rate * abs(notional)
            maintenance_required = maint_margin_rate * abs(notional)
            
            # Margin call check
            margin_call_occurred = False
            shortfall = 0
            if equity < maintenance_required:
                shortfall = initial_required - equity
                if shortfall > 0:
                    debt += shortfall
                    equity += shortfall
                    margin_calls_count += 1
                    margin_call_occurred = True
                    
                    if equity <= 0:
                        results['failures'] += 1
                        failed = True
                        failure_day = day
                        break
            
            # Interest calculation
            daily_interest = 0
            if debt > 0:
                daily_interest = debt * funding_daily
                debt += daily_interest
                equity -= daily_interest
                total_interest += daily_interest
                
                if equity <= 0:
                    results['failures'] += 1
                    failed = True
                    failure_day = day
                    break
            
            # Debt repayment
            repayment = 0
            if debt > 0 and equity > initial_required * 1.1:
                repayment = min(debt, equity - initial_required)
                debt -= repayment
                equity -= repayment
            
            # Update maximum drawdown
            if equity > peak_equity:
                peak_equity = equity
            current_drawdown = (peak_equity - equity) / peak_equity
            max_drawdown = max(max_drawdown, current_drawdown)
            
            # Store plotting data
            if sim in plot_indices and day % 5 == 0:
                equity_path.append(equity)
                days_path.append(day)
            
            # Store detailed data
            if record_details:
                daily_records.append({
                    'day': day,
                    'year': day / trading_days,
                    'notional': notional,
                    'effective_leverage': notional / equity if equity > 0 else float('inf'),
                    'daily_return_pct': daily_return * 100,
                    'daily_pnl': daily_pnl,
                    'equity_before': equity_before,
                    'equity_after': equity,
                    'debt': debt,
                    'margin_call': margin_call_occurred,
                    'rebalancing': is_rebalancing_day,
                    'rebalancing_cost': rebalancing_cost,
                    'daily_interest': daily_interest,
                    'drawdown_pct': current_drawdown * 100
                })
        
        # Store results for completed paths
        if not failed:
            results['final_equities'].append(equity)
            results['max_drawdowns'].append(max_drawdown)
            results['total_interests'].append(total_interest)
            results['total_rebalancing_costs'].append(total_rebalancing_cost)
            
            if margin_calls_count > 0:
                results['margin_call_paths'] += 1
                results['total_margin_calls'] += margin_calls_count
        
        # Store sample path for plotting
        if sim in plot_indices:
            results['sample_paths'].append({
                'days': days_path,
                'equity': equity_path,
                'failed': failed,
                'margin_calls': margin_calls_count
            })
        
        # Store detailed path for Excel
        if record_details:
            results['detailed_paths'][sim] = {
                'simulation_id': sim + 1,
                'failed': failed,
                'failure_day': failure_day,
                'final_equity': equity if not failed else 0,
                'max_drawdown_pct': max_drawdown * 100,
                'margin_calls_count': margin_calls_count,
                'total_interest': total_interest,
                'total_rebalancing_cost': total_rebalancing_cost,
                'daily_data': daily_records,
                'rebalancing_events': rebalancing_events
            }
        
        # Store path summary
        results['path_summaries'].append({
            'simulation_id': sim + 1,
            'failed': failed,
            'failure_day': failure_day,
            'final_equity': equity if not failed else 0,
            'max_drawdown_pct': max_drawdown * 100,
            'margin_calls_count': margin_calls_count,
            'total_interest': total_interest,
            'total_rebalancing_cost': total_rebalancing_cost,
            'final_return_pct': ((equity / initial_capital) - 1) * 100 if not failed else -100
        })
        
        # Progress reporting
        if (sim + 1) % 1000 == 0:
            failure_rate = results['failures'] / (sim + 1) * 100
            avg_rebal_cost = np.mean(results['total_rebalancing_costs']) if results['total_rebalancing_costs'] else 0
            print(f"Progress: {sim+1:,}/{simulations:,} - "
                  f"Failure rate: {failure_rate:.3f}% - "
                  f"Avg rebalancing cost: {avg_rebal_cost:.2f}")
    
    return results

def create_comprehensive_plots(results_16x, results_24x):
    """Create 9-panel dashboard matching the user's attached image"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Portfolio Paths - Leverage 1.6x
    ax1 = plt.subplot(3, 3, 1)
    for path in results_16x['sample_paths'][:25]:
        color = 'red' if path['failed'] else 'blue'
        alpha = 0.8 if path['failed'] else 0.4
        years_axis = np.array(path['days']) / 252
        plt.plot(years_axis, path['equity'], color=color, alpha=alpha, linewidth=1)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.axhline(y=21.92, color='green', linestyle=':', alpha=0.7, label='Initial Margin')
    plt.axhline(y=19.52, color='red', linestyle='-.', alpha=0.7, label='Maintenance Margin')
    plt.title('Leverage 1.6x - Portfolio Paths', fontweight='bold')
    plt.xlabel('Years')
    plt.ylabel('Equity')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 2. Portfolio Paths - Leverage 2.4x
    ax2 = plt.subplot(3, 3, 2)
    for path in results_24x['sample_paths'][:25]:
        color = 'red' if path['failed'] else 'orange'
        alpha = 0.8 if path['failed'] else 0.4
        years_axis = np.array(path['days']) / 252
        plt.plot(years_axis, path['equity'], color=color, alpha=alpha, linewidth=1)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.axhline(y=32.88, color='green', linestyle=':', alpha=0.7, label='Initial Margin')
    plt.axhline(y=29.28, color='red', linestyle='-.', alpha=0.7, label='Maintenance Margin')
    plt.title('Leverage 2.4x - Portfolio Paths', fontweight='bold')
    plt.xlabel('Years')
    plt.ylabel('Equity')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 3. 5-Year Failure Probability
    ax3 = plt.subplot(3, 3, 3)
    leverages = ['1.6x', '2.4x']
    failure_rates = [
        results_16x['failures'] / 10000 * 100,
        results_24x['failures'] / 10000 * 100
    ]
    bars = plt.bar(leverages, failure_rates, color=['skyblue', 'salmon'], alpha=0.8)
    plt.title('5-Year Failure Probability', fontweight='bold')
    plt.ylabel('Failure Probability (%)')
    
    for i, v in enumerate(failure_rates):
        plt.text(i, v + max(0.001, v*0.1), f'{v:.3f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 4. Final Equity Distribution
    ax4 = plt.subplot(3, 3, 4)
    if results_16x['final_equities']:
        plt.hist(results_16x['final_equities'], bins=50, alpha=0.7, 
                label=f'1.6x (n={len(results_16x["final_equities"]):,})', 
                color='skyblue', density=True)
    if results_24x['final_equities']:
        plt.hist(results_24x['final_equities'], bins=50, alpha=0.7, 
                label=f'2.4x (n={len(results_24x["final_equities"]):,})', 
                color='salmon', density=True)
    
    plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
    plt.title('Final Equity Distribution', fontweight='bold')
    plt.xlabel('Final Equity')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Maximum Drawdown Distribution
    ax5 = plt.subplot(3, 3, 5)
    if results_16x['max_drawdowns']:
        mdd_16x = [dd * 100 for dd in results_16x['max_drawdowns']]
        plt.hist(mdd_16x, bins=40, alpha=0.7, label='1.6x', color='skyblue', density=True)
    if results_24x['max_drawdowns']:
        mdd_24x = [dd * 100 for dd in results_24x['max_drawdowns']]
        plt.hist(mdd_24x, bins=40, alpha=0.7, label='2.4x', color='salmon', density=True)
    
    plt.axvline(x=32, color='red', linestyle='--', linewidth=2, label='Historical MDD 32%')
    plt.title('Maximum Drawdown Distribution', fontweight='bold')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Margin Call Experience Rate
    ax6 = plt.subplot(3, 3, 6)
    margin_rates = [
        results_16x['margin_call_paths'] / 10000 * 100,
        results_24x['margin_call_paths'] / 10000 * 100
    ]
    bars = plt.bar(leverages, margin_rates, color=['lightblue', 'lightcoral'], alpha=0.8)
    plt.title('Margin Call Experience Rate', fontweight='bold')
    plt.ylabel('Margin Call Rate (%)')
    
    for i, v in enumerate(margin_rates):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 7. Cumulative Interest Distribution
    ax7 = plt.subplot(3, 3, 7)
    if results_16x['total_interests']:
        plt.hist(results_16x['total_interests'], bins=30, alpha=0.7, 
                label='1.6x', color='skyblue', density=True)
    if results_24x['total_interests']:
        plt.hist(results_24x['total_interests'], bins=30, alpha=0.7, 
                label='2.4x', color='salmon', density=True)
    
    plt.title('Cumulative Interest Distribution', fontweight='bold')
    plt.xlabel('Total Interest Paid')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Risk vs Return Profile
    ax8 = plt.subplot(3, 3, 8)
    if results_16x['final_equities'] and results_16x['max_drawdowns']:
        returns_16x = [(eq/100 - 1) * 100 for eq in results_16x['final_equities']]
        mdd_16x = [dd * 100 for dd in results_16x['max_drawdowns']]
        plt.scatter(mdd_16x, returns_16x, alpha=0.5, s=8, color='blue', label='1.6x')
    
    if results_24x['final_equities'] and results_24x['max_drawdowns']:
        returns_24x = [(eq/100 - 1) * 100 for eq in results_24x['final_equities']]
        mdd_24x = [dd * 100 for dd in results_24x['max_drawdowns']]
        plt.scatter(mdd_24x, returns_24x, alpha=0.5, s=8, color='red', label='2.4x')
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.axvline(x=32, color='red', linestyle='--', alpha=0.5, label='Historical MDD')
    plt.title('Risk vs Return Profile', fontweight='bold')
    plt.xlabel('Maximum Drawdown (%)')
    plt.ylabel('Total Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    stats_data = []
    for name, results in [('1.6x', results_16x), ('2.4x', results_24x)]:
        failure_rate = results['failures'] / 10000 * 100
        margin_rate = results['margin_call_paths'] / 10000 * 100
        avg_final = np.mean(results['final_equities']) if results['final_equities'] else 0
        avg_mdd = np.mean(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0
        avg_interest = np.mean(results['total_interests']) if results['total_interests'] else 0
        
        stats_data.append([
            name,
            f"{failure_rate:.3f}%",
            f"{margin_rate:.1f}%",
            f"{avg_final:.0f}",
            f"{avg_mdd:.1f}%",
            f"{avg_interest:.2f}"
        ])
    
    table = ax9.table(
        cellText=stats_data,
        colLabels=['Leverage', 'Failure Rate', 'MC Rate', 'Avg Final', 'Avg MDD', 'Avg Interest'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax9.set_title('Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

def export_results_to_excel(results, leverage, filename_prefix="rebalancing_simulation"):
    """Export comprehensive simulation results to Excel"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_leverage_{leverage}x_{timestamp}.xlsx"
    
    print(f"\nExporting results to: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Summary Statistics
        summary_stats = {
            'Metric': [
                'Total Simulations', 'Leverage Ratio', 'Simulation Period (Years)',
                'Rebalancing Type', 'Failure Count', 'Failure Rate (%)', 'Survival Rate (%)',
                'Margin Call Paths', 'Margin Call Rate (%)',
                'Avg Final Equity (Survivors)', 'Median Final Equity (Survivors)',
                'Avg Total Return (Survivors)', 'Avg Max Drawdown (%)',
                'Worst Max Drawdown (%)', 'Avg Total Interest Paid',
                'Avg Total Rebalancing Cost', 'Paths Exported for Analysis'
            ],
            'Value': [
                10000, f"{leverage}x", 5, "20-Day Rebalancing",
                results['failures'], results['failures'] / 10000 * 100,
                (10000 - results['failures']) / 10000 * 100,
                results['margin_call_paths'], results['margin_call_paths'] / 10000 * 100,
                np.mean(results['final_equities']) if results['final_equities'] else 0,
                np.median(results['final_equities']) if results['final_equities'] else 0,
                np.mean([(eq/100-1)*100 for eq in results['final_equities']]) if results['final_equities'] else 0,
                np.mean(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0,
                np.max(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0,
                np.mean(results['total_interests']) if results['total_interests'] else 0,
                np.mean(results['total_rebalancing_costs']) if results['total_rebalancing_costs'] else 0,
                len(results['detailed_paths'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: All Path Summaries
        all_paths_df = pd.DataFrame(results['path_summaries'])
        all_paths_df.to_excel(writer, sheet_name='All_Paths_Summary', index=False)
        
        # Sheet 3: Detailed Path Overview
        detailed_overview = []
        for sim_id, path_data in results['detailed_paths'].items():
            detailed_overview.append({
                'Simulation_ID': path_data['simulation_id'],
                'Final_Status': 'Failed' if path_data['failed'] else 'Survived',
                'Final_Equity': round(path_data['final_equity'], 2),
                'Final_Return_Pct': round(((path_data['final_equity']/100) - 1) * 100, 2),
                'Max_Drawdown_Pct': round(path_data['max_drawdown_pct'], 2),
                'Margin_Calls': path_data['margin_calls_count'],
                'Total_Interest': round(path_data['total_interest'], 2),
                'Total_Rebalancing_Cost': round(path_data['total_rebalancing_cost'], 2),
                'Rebalancing_Events': len(path_data['rebalancing_events']),
                'Failure_Day': path_data['failure_day'] if path_data['failed'] else 'N/A'
            })
        
        if detailed_overview:
            detailed_df = pd.DataFrame(detailed_overview)
            detailed_df.to_excel(writer, sheet_name='Detailed_Paths_Overview', index=False)
        
        # Sheet 4: Daily Sample Data (first 50 paths)
        daily_sample_data = []
        sample_paths = list(results['detailed_paths'].items())[:50]
        
        for sim_id, path_data in sample_paths:
            for record in path_data['daily_data']:
                daily_sample_data.append({
                    'Simulation_ID': path_data['simulation_id'],
                    'Day': record['day'],
                    'Year': round(record['year'], 3),
                    'Notional': round(record['notional'], 2),
                    'Effective_Leverage': round(record['effective_leverage'], 3),
                    'Daily_Return_Pct': round(record['daily_return_pct'], 4),
                    'Daily_PnL': round(record['daily_pnl'], 2),
                    'Equity_Before': round(record['equity_before'], 2),
                    'Equity_After': round(record['equity_after'], 2),
                    'Debt': round(record['debt'], 2),
                    'Margin_Call': record['margin_call'],
                    'Rebalancing': record['rebalancing'],
                    'Rebalancing_Cost': round(record['rebalancing_cost'], 4),
                    'Daily_Interest': round(record['daily_interest'], 4),
                    'Drawdown_Pct': round(record['drawdown_pct'], 2)
                })
        
        if daily_sample_data:
            daily_df = pd.DataFrame(daily_sample_data)
            daily_df.to_excel(writer, sheet_name='Daily_Sample_Data', index=False)
    
    print(f"✅ Excel export completed: {filename}")
    return filename

def print_results_summary(name, results):
    """Print comprehensive results summary"""
    print(f"\n{'='*60}")
    print(f"{name} Simulation Results (20-Day Rebalancing)")
    print(f"{'='*60}")
    
    failure_rate = results['failures'] / 10000 * 100
    margin_rate = results['margin_call_paths'] / 10000 * 100
    
    print(f"Basic Statistics:")
    print(f"   • 5-year failure probability: {failure_rate:.3f}%")
    print(f"   • Survival rate: {100 - failure_rate:.3f}%")
    print(f"   • Margin call experience rate: {margin_rate:.2f}%")
    
    if results['final_equities']:
        final_eq = results['final_equities']
        returns = [(eq/100 - 1) * 100 for eq in final_eq]
        print(f"\nSurvivor Performance:")
        print(f"   • Mean final equity: {np.mean(final_eq):.1f}")
        print(f"   • Mean return: {np.mean(returns):.1f}%")
        print(f"   • Positive return rate: {sum(1 for r in returns if r > 0)/len(returns)*100:.1f}%")
    
    if results['max_drawdowns']:
        mdd_data = [dd * 100 for dd in results['max_drawdowns']]
        print(f"\nDrawdown Statistics:")
        print(f"   • Mean MDD: {np.mean(mdd_data):.1f}%")
        print(f"   • Worst MDD: {np.max(mdd_data):.1f}%")
    
    if results['total_rebalancing_costs']:
        rebal_data = results['total_rebalancing_costs']
        print(f"\nRebalancing Costs:")
        print(f"   • Mean total cost: {np.mean(rebal_data):.2f}")
        print(f"   • Max total cost: {np.max(rebal_data):.2f}")

def run_complete_analysis():
    """Execute complete simulation with both plots and Excel export"""
    
    print("🔄 20-Day Rebalancing Monte Carlo Simulation")
    print("📊 Parameters: Annual Return 10%, Annual Volatility 10%, Margins 13.7%/12.2%")
    
    # Common parameters
    common_params = {
        'mu_annual': 0.10,
        'sigma_annual': 0.10,
        'init_margin_rate': 0.137,
        'maint_margin_rate': 0.122,
        'funding_rate_annual': 0.068,
        'years': 5,
        'simulations': 10000,
        'export_sample_size': 1000,
        'rebalancing_frequency': 20
    }
    
    # Run simulations
    print("\n1️⃣ Running Leverage 1.6x Simulation...")
    results_16x = futures_portfolio_simulation_with_rebalancing(target_leverage=1.6, **common_params)
    
    print("\n2️⃣ Running Leverage 2.4x Simulation...")
    results_24x = futures_portfolio_simulation_with_rebalancing(target_leverage=2.4, **common_params)
    
    # Print results
    print_results_summary("Leverage 1.6x", results_16x)
    print_results_summary("Leverage 2.4x", results_24x)
    
    # Generate plots (THIS CREATES THE VISUALIZATION)
    print(f"\n📊 Creating comprehensive visualization charts...")
    create_comprehensive_plots(results_16x, results_24x)
    
    # Export to Excel
    file_16x = export_results_to_excel(results_16x, 1.6)
    file_24x = export_results_to_excel(results_24x, 2.4)
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 Analysis Complete")
    print(f"{'='*60}")
    print(f"📊 Plots: Displayed on screen (9-panel dashboard)")
    print(f"📁 Excel files: {file_16x}, {file_24x}")
    print(f"📋 Each Excel file contains 1,000 detailed simulation paths")
    
    return results_16x, results_24x, file_16x, file_24x

# Execute the complete analysis
if __name__ == "__main__":
    results_16x, results_24x, excel_16x, excel_24x = run_complete_analysis()
