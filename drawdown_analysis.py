import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import sys
import argparse

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
    export_sample_size=1000
):
    """
    Monte Carlo simulation with 20-day rebalancing to target leverage.
    Funding costs apply only when margin calls occur.
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
    print(f"- Annual return: {mu_annual:.1%}, Annual volatility: {sigma_annual:.1%}")
    
    # Results storage
    results = {
        'failures': 0,
        'margin_call_paths': 0,
        'total_margin_calls': 0,
        'final_equities': [],
        'max_drawdowns': [],
        'total_interests': [],
        'total_rebalancing_costs': [],
        'sample_paths': [],
        'detailed_paths': {},
        'path_summaries': []
    }
    
    # Select paths for detailed tracking
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
            is_rebalancing_day = (day % rebalancing_frequency == 0)
            
            # Generate daily return with t-distribution (variance normalized)
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
            
            # Interest calculation (only when debt exists)
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

def extract_exact_mdd_statistics(leverage_results):
    """Extract exact single-number MDD statistics from multiple leverage simulation results"""
    
    print("\n" + "="*70)
    print("EXACT MDD STATISTICS FROM 10,000 SIMULATIONS")
    print("="*70)
    
    stats_dict = {}
    
    for leverage, results in leverage_results.items():
        if results['max_drawdowns']:
            mdd_data = np.array(results['max_drawdowns']) * 100
            
            stats = {
                'count': len(mdd_data),
                'median': np.percentile(mdd_data, 50),
                'p95': np.percentile(mdd_data, 95),
                'worst': np.max(mdd_data),
                'mean': np.mean(mdd_data)
            }
            
            historical_percentile = (np.sum(mdd_data <= 32) / len(mdd_data)) * 100
            
            print(f"\nLEVERAGE {leverage}x - EXACT MDD NUMBERS:")
            print(f"   • Sample Size: {stats['count']:,} surviving portfolios")
            print(f"   • Median MDD: {stats['median']:.2f}%")
            print(f"   • 95th Percentile MDD: {stats['p95']:.2f}%")
            print(f"   • Worst MDD: {stats['worst']:.2f}%")
            print(f"   • Mean MDD: {stats['mean']:.2f}%")
            print(f"   • Historical 32% MDD is at {historical_percentile:.1f}th percentile")
            
            stats_dict[f'{leverage}x'] = stats
    
    return stats_dict

def create_enhanced_plots_with_save(leverage_results):
    """Create improved dashboard with separate subplots for each leverage and only required charts"""
    
    leverages = list(leverage_results.keys())
    num_leverages = len(leverages)
    
    # Create figure with dynamic sizing based on number of leverages
    fig = plt.figure(figsize=(5*num_leverages, 16))
    
    # Adjust subplot spacing
    plt.subplots_adjust(
        left=0.06, bottom=0.06, right=0.96, top=0.94,
        wspace=0.25, hspace=0.35
    )
    
    # Define colors for each leverage
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    # 1. Portfolio Paths - One subplot per leverage
    for i, (leverage, results) in enumerate(leverage_results.items()):
        ax = plt.subplot(4, num_leverages, i+1)
        
        for path in results['sample_paths'][:25]:
            path_color = 'red' if path['failed'] else colors[i % len(colors)]
            alpha = 0.8 if path['failed'] else 0.4
            years_axis = np.array(path['days']) / 252
            plt.plot(years_axis, path['equity'], color=path_color, alpha=alpha, linewidth=1)
        
        # Calculate margin requirements for this leverage
        init_margin = 0.137 * leverage * 100
        maint_margin = 0.122 * leverage * 100
        
        plt.axhline(y=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.axhline(y=init_margin, color='green', linestyle=':', alpha=0.7, label='Initial Margin')
        plt.axhline(y=maint_margin, color='red', linestyle='-.', alpha=0.7, label='Maintenance Margin')
        plt.title(f'Leverage {leverage}x - Portfolio Paths', fontweight='bold', fontsize=12)
        plt.xlabel('Years')
        plt.ylabel('Equity')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
    
    # 2. Final Equity Distribution - One subplot per leverage
    for i, (leverage, results) in enumerate(leverage_results.items()):
        ax = plt.subplot(4, num_leverages, num_leverages + i + 1)
        
        if results['final_equities']:
            filtered_equities = [eq for eq in results['final_equities'] if eq <= 1500]
            plt.hist(filtered_equities, bins=50, alpha=0.7, 
                    color=colors[i % len(colors)], density=True,
                    label=f'{leverage}x (n={len(results["final_equities"]):,})')
        
        plt.axvline(x=100, color='black', linestyle='--', alpha=0.7, label='Initial Capital')
        plt.xlim(0, 1500)
        plt.title(f'Leverage {leverage}x - Final Equity Distribution', fontweight='bold', fontsize=12)
        plt.xlabel('Final Equity')
        plt.ylabel('Density')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
    
    # 3. Maximum Drawdown Distribution - One subplot per leverage
    for i, (leverage, results) in enumerate(leverage_results.items()):
        ax = plt.subplot(4, num_leverages, 2*num_leverages + i + 1)
        
        if results['max_drawdowns']:
            mdd_data = [dd * 100 for dd in results['max_drawdowns']]
            plt.hist(mdd_data, bins=40, alpha=0.7, 
                    color=colors[i % len(colors)], density=True,
                    label=f'{leverage}x')
        
        plt.axvline(x=32, color='red', linestyle='--', linewidth=2, label='Historical MDD 32%')
        plt.title(f'Leverage {leverage}x - Maximum Drawdown Distribution', fontweight='bold', fontsize=12)
        plt.xlabel('Maximum Drawdown (%)')
        plt.ylabel('Density')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
    
    # 4. Risk vs Return Profile - One subplot per leverage
    for i, (leverage, results) in enumerate(leverage_results.items()):
        ax = plt.subplot(4, num_leverages, 3*num_leverages + i + 1)
        
        if results['final_equities'] and results['max_drawdowns']:
            returns = [(eq/100 - 1) * 100 for eq in results['final_equities'] if eq <= 1500]
            mdd_filtered = [results['max_drawdowns'][j] * 100 
                           for j, eq in enumerate(results['final_equities']) if eq <= 1500]
            plt.scatter(mdd_filtered, returns, alpha=0.5, s=8, 
                       color=colors[i % len(colors)], label=f'{leverage}x')
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=32, color='red', linestyle='--', alpha=0.5, label='Historical MDD')
        plt.ylim(-100, 1400)
        plt.title(f'Leverage {leverage}x - Risk vs Return Profile', fontweight='bold', fontsize=12)
        plt.xlabel('Maximum Drawdown (%)')
        plt.ylabel('Total Return (%)')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
    
    # Show the plot
    plt.show()
    
    # Save as high-quality PNG file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    png_filename = f"rebalancing_simulation_dashboard_{timestamp}.png"
    
    fig.savefig(
        png_filename, 
        dpi=300,
        bbox_inches='tight',
        facecolor='white',
        edgecolor='none',
        format='png'
    )
    
    print(f"\nDashboard plot saved as: {png_filename}")
    return png_filename

def export_complete_mdd_data_to_excel(leverage_results):
    """Export all MDD values to Excel for verification"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Complete_MDD_Analysis_{timestamp}.xlsx"
    
    print(f"\nExporting complete MDD data to: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Summary Statistics (Single Numbers)
        summary_data = []
        
        for leverage, results in leverage_results.items():
            if results['max_drawdowns']:
                mdd_data = np.array(results['max_drawdowns']) * 100
                summary_data.append({
                    'Leverage': f'{leverage}x',
                    'Sample_Size': len(mdd_data),
                    'Median_MDD': round(np.percentile(mdd_data, 50), 2),
                    'P95_MDD': round(np.percentile(mdd_data, 95), 2),
                    'Worst_MDD': round(np.max(mdd_data), 2),
                    'Mean_MDD': round(np.mean(mdd_data), 2),
                    'Historical_32pct_Percentile': round((np.sum(mdd_data <= 32) / len(mdd_data)) * 100, 1)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='MDD_Summary_Statistics', index=False)
        
        # Sheets for each leverage: All MDD Values (COMPLETE LIST)
        for leverage, results in leverage_results.items():
            if results['max_drawdowns']:
                all_mdd = pd.DataFrame({
                    'Simulation_ID': range(1, len(results['max_drawdowns']) + 1),
                    'Max_Drawdown_Pct': [round(dd * 100, 3) for dd in results['max_drawdowns']]
                })
                all_mdd = all_mdd.sort_values('Max_Drawdown_Pct', ascending=False).reset_index(drop=True)
                all_mdd['Rank'] = range(1, len(all_mdd) + 1)
                
                sheet_name = f'All_{leverage}x_MDD_Values'
                # Excel sheet names cannot exceed 31 characters
                if len(sheet_name) > 31:
                    sheet_name = f'MDD_{leverage}x_Values'
                
                all_mdd.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Sheet: Percentile Breakdown for all leverages
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        percentile_data = []
        
        for p in percentiles:
            row = {'Percentile': f'{p}th'}
            
            for leverage, results in leverage_results.items():
                if results['max_drawdowns']:
                    mdd_data = np.array(results['max_drawdowns']) * 100
                    column_name = f'Leverage_{leverage}x_MDD'
                    if p == 100:
                        row[column_name] = round(np.max(mdd_data), 2)
                    else:
                        row[column_name] = round(np.percentile(mdd_data, p), 2)
            
            percentile_data.append(row)
        
        percentile_df = pd.DataFrame(percentile_data)
        percentile_df.to_excel(writer, sheet_name='Percentile_Breakdown', index=False)
    
    print(f"Complete MDD data exported: {filename}")
    return filename

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
    
    print(f"Excel export completed: {filename}")
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

def parse_arguments():
    """Parse command line arguments for leverage values"""
    parser = argparse.ArgumentParser(
        description='Monte Carlo simulation for leveraged portfolio analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python drawdown_analysis.py                    # Default: 1.6x, 2.0x, 2.4x
  python drawdown_analysis.py 3.0               # 1.6x, 2.0x, 2.4x, 3.0x
  python drawdown_analysis.py 2.8 3.2           # 1.6x, 2.0x, 2.4x, 2.8x, 3.2x
        """
    )
    
    parser.add_argument(
        'additional_leverages', 
        nargs='*', 
        type=float,
        help='Additional leverage ratios (up to 2 additional values, max 5 total)'
    )
    
    args = parser.parse_args()
    
    # Default leverages
    default_leverages = [1.6, 2.0, 2.4]
    
    # Validate additional leverages
    if len(args.additional_leverages) > 2:
        print("Error: Maximum 2 additional leverage values allowed (5 total maximum)")
        sys.exit(1)
    
    # Validate leverage values are positive
    for lev in args.additional_leverages:
        if lev <= 0:
            print(f"Error: Leverage values must be positive, got {lev}")
            sys.exit(1)
        if lev > 10:
            print(f"Warning: Very high leverage {lev}x detected. This may cause extreme results.")
    
    # Combine default and additional leverages
    all_leverages = default_leverages + list(args.additional_leverages)
    
    # Remove duplicates while preserving order
    unique_leverages = []
    for lev in all_leverages:
        if lev not in unique_leverages:
            unique_leverages.append(lev)
    
    return sorted(unique_leverages)

def run_complete_simulation(leverages=None):
    """Execute complete simulation with all requested features"""
    
    if leverages is None:
        leverages = [1.6, 2.0, 2.4]  # Default leverages
    
    print("20-Day Rebalancing Monte Carlo Simulation")
    print("Parameters: Annual Return 10%, Annual Volatility 10%, Margins 13.7%/12.2%")
    print(f"Analyzing leverages: {[f'{l}x' for l in leverages]}")
    
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
    
    # Run simulations for all leverages
    leverage_results = {}
    excel_files = []
    
    for i, leverage in enumerate(leverages, 1):
        print(f"\n{i}️⃣ Running Leverage {leverage}x Simulation...")
        results = futures_portfolio_simulation_with_rebalancing(target_leverage=leverage, **common_params)
        leverage_results[leverage] = results
        
        # Print results
        print_results_summary(f"Leverage {leverage}x", results)
        
        # Export detailed simulation results
        excel_file = export_results_to_excel(results, leverage)
        excel_files.append(excel_file)
    
    # Extract exact MDD statistics
    mdd_stats = extract_exact_mdd_statistics(leverage_results)
    
    # Export complete MDD data
    mdd_excel_file = export_complete_mdd_data_to_excel(leverage_results)
    
    # Create and save enhanced plots
    print(f"\nCreating enhanced visualization dashboard...")
    png_file = create_enhanced_plots_with_save(leverage_results)
    
    # Final summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Generated Files:")
    print(f"   📊 Dashboard PNG: {png_file}")
    print(f"   📋 Complete MDD Analysis: {mdd_excel_file}")
    print(f"   📄 Detailed Results: {', '.join(excel_files)}")
    print(f"\nAll outputs in English as requested")
    
    return leverage_results, mdd_stats

# Execute the complete analysis
if __name__ == "__main__":
    # Parse command line arguments
    leverages = parse_arguments()
    
    print(f"Starting simulation with leverages: {leverages}")
    
    # Run the complete simulation
    leverage_results, mdd_statistics = run_complete_simulation(leverages)
    
    print("\n✅ Complete simulation analysis finished successfully!")
