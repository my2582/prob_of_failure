import numpy as np
import pandas as pd
from scipy import stats
import random
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def futures_portfolio_simulation_with_export(
    initial_capital=100.0,
    leverage=1.6,
    mu_annual=0.10,
    sigma_annual=0.10,
    years=5,
    trading_days=252,
    init_margin_rate=0.137,
    maint_margin_rate=0.122,
    funding_rate_annual=0.068,
    t_df=4,
    simulations=10000,
    export_sample_size=1000,
    transaction_cost_bp=3,
    random_seed=42
):
    """
    Enhanced Monte Carlo simulation with selective path recording for Excel export
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Calculate parameters
    notional = leverage * initial_capital
    mu_daily = mu_annual / trading_days
    sigma_daily = sigma_annual / np.sqrt(trading_days)
    funding_daily = funding_rate_annual / trading_days
    transaction_daily = transaction_cost_bp / 10000 / trading_days
    initial_required = init_margin_rate * notional
    maintenance_required = maint_margin_rate * notional
    total_days = trading_days * years
    
    # Randomly select which simulations to record in detail
    export_indices = set(random.sample(range(simulations), export_sample_size))
    
    print(f"Simulation Configuration - Leverage {leverage}x:")
    print(f"- Notional exposure: {notional:.1f}")
    print(f"- Initial margin: {initial_required:.1f} ({init_margin_rate:.1%})")
    print(f"- Maintenance margin: {maintenance_required:.1f} ({maint_margin_rate:.1%})")
    print(f"- Recording {len(export_indices)} detailed paths for export")
    
    # Results storage
    results = {
        'failures': 0,
        'margin_call_paths': 0,
        'total_margin_calls': 0,
        'final_equities': [],
        'max_drawdowns': [],
        'total_interests': [],
        'detailed_paths': {},  # Only store selected paths
        'path_summaries': []   # Store summary for all paths
    }
    
    # Run simulations
    for sim in range(simulations):
        # Initialize simulation variables
        equity = initial_capital
        debt = 0.0
        peak_equity = initial_capital
        max_drawdown = 0.0
        margin_calls_count = 0
        total_interest = 0.0
        failed = False
        failure_day = None
        
        # Detailed recording for selected paths
        record_details = sim in export_indices
        if record_details:
            daily_records = []
        
        # Daily simulation loop
        for day in range(1, total_days + 1):
            # Generate return with t-distribution
            z = stats.t.rvs(df=t_df)
            z *= np.sqrt((t_df - 2) / t_df)  # Variance normalization
            daily_return = mu_daily + sigma_daily * z
            
            # Calculate P&L and costs
            daily_pnl = notional * daily_return
            daily_cost = notional * transaction_daily
            equity_before = equity
            equity += daily_pnl - daily_cost
            
            # Check immediate failure
            if equity <= 0:
                results['failures'] += 1
                failed = True
                failure_day = day
                break
            
            # Margin call mechanics
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
            
            # Debt repayment with excess funds
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
            
            # Record detailed data for selected paths
            if record_details:
                daily_records.append({
                    'day': day,
                    'year': day / trading_days,
                    'daily_return_pct': daily_return * 100,
                    'daily_pnl': daily_pnl,
                    'equity_before': equity_before,
                    'equity_after': equity,
                    'debt': debt,
                    'margin_call': margin_call_occurred,
                    'shortfall': shortfall,
                    'daily_interest': daily_interest,
                    'repayment': repayment,
                    'drawdown_pct': current_drawdown * 100,
                    'peak_equity': peak_equity
                })
        
        # Store results
        if not failed:
            results['final_equities'].append(equity)
            results['max_drawdowns'].append(max_drawdown)
            results['total_interests'].append(total_interest)
            
            if margin_calls_count > 0:
                results['margin_call_paths'] += 1
                results['total_margin_calls'] += margin_calls_count
        
        # Store detailed path data for selected simulations
        if record_details:
            results['detailed_paths'][sim] = {
                'simulation_id': sim + 1,
                'failed': failed,
                'failure_day': failure_day,
                'final_equity': equity if not failed else 0,
                'max_drawdown_pct': max_drawdown * 100,
                'margin_calls_count': margin_calls_count,
                'total_interest': total_interest,
                'daily_data': daily_records
            }
        
        # Store summary for all paths
        results['path_summaries'].append({
            'simulation_id': sim + 1,
            'failed': failed,
            'failure_day': failure_day,
            'final_equity': equity if not failed else 0,
            'max_drawdown_pct': max_drawdown * 100,
            'margin_calls_count': margin_calls_count,
            'total_interest': total_interest,
            'final_return_pct': ((equity / initial_capital) - 1) * 100 if not failed else -100
        })
        
        # Progress reporting
        if (sim + 1) % 1000 == 0:
            failure_rate = results['failures'] / (sim + 1) * 100
            print(f"Progress: {sim+1:,}/{simulations:,} - Current failure rate: {failure_rate:.3f}%")
    
    return results

def export_simulation_results_to_excel(results, leverage, filename_prefix="monte_carlo_paths"):
    """
    Export simulation results to Excel with comprehensive analysis sheets
    """
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_leverage_{leverage}x_{timestamp}.xlsx"
    
    print(f"\nExporting results to: {filename}")
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # Sheet 1: Simulation Summary
        summary_stats = {
            'Metric': [
                'Total Simulations', 'Leverage Ratio', 'Simulation Period (Years)',
                'Failure Count', 'Failure Rate (%)', 'Survival Rate (%)',
                'Margin Call Paths', 'Margin Call Rate (%)',
                'Avg Final Equity (Survivors)', 'Median Final Equity (Survivors)',
                'Avg Total Return (Survivors)', 'Avg Max Drawdown (%)',
                'Worst Max Drawdown (%)', 'Avg Total Interest Paid',
                'Paths Exported for Analysis'
            ],
            'Value': [
                10000, f"{leverage}x", 5,
                results['failures'], 
                results['failures'] / 10000 * 100,
                (10000 - results['failures']) / 10000 * 100,
                results['margin_call_paths'],
                results['margin_call_paths'] / 10000 * 100,
                np.mean(results['final_equities']) if results['final_equities'] else 0,
                np.median(results['final_equities']) if results['final_equities'] else 0,
                np.mean([(eq/100-1)*100 for eq in results['final_equities']]) if results['final_equities'] else 0,
                np.mean(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0,
                np.max(results['max_drawdowns']) * 100 if results['max_drawdowns'] else 0,
                np.mean(results['total_interests']) if results['total_interests'] else 0,
                len(results['detailed_paths'])
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Sheet 2: All Path Summaries (10,000 paths overview)
        all_paths_df = pd.DataFrame(results['path_summaries'])
        all_paths_df.to_excel(writer, sheet_name='All_Paths_Summary', index=False)
        
        # Sheet 3: Detailed Path Overview (1,000 exported paths)
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
                'Failure_Day': path_data['failure_day'] if path_data['failed'] else 'N/A',
                'Days_Survived': len(path_data['daily_data']) if not path_data['failed'] else path_data['failure_day']
            })
        
        detailed_df = pd.DataFrame(detailed_overview)
        detailed_df.to_excel(writer, sheet_name='Exported_Paths_Overview', index=False)
        
        # Sheet 4: Daily Data Sample (first 50 paths to manage file size)
        daily_sample_data = []
        sample_paths = list(results['detailed_paths'].items())[:50]
        
        for sim_id, path_data in sample_paths:
            for record in path_data['daily_data']:
                daily_sample_data.append({
                    'Simulation_ID': path_data['simulation_id'],
                    'Day': record['day'],
                    'Year': round(record['year'], 3),
                    'Daily_Return_Pct': round(record['daily_return_pct'], 4),
                    'Daily_PnL': round(record['daily_pnl'], 2),
                    'Equity_Before': round(record['equity_before'], 2),
                    'Equity_After': round(record['equity_after'], 2),
                    'Debt': round(record['debt'], 2),
                    'Margin_Call': record['margin_call'],
                    'Shortfall': round(record['shortfall'], 2),
                    'Daily_Interest': round(record['daily_interest'], 4),
                    'Repayment': round(record['repayment'], 2),
                    'Drawdown_Pct': round(record['drawdown_pct'], 2),
                    'Peak_Equity': round(record['peak_equity'], 2)
                })
        
        if daily_sample_data:
            daily_df = pd.DataFrame(daily_sample_data)
            daily_df.to_excel(writer, sheet_name='Daily_Sample_Data', index=False)
        
        # Sheet 5: Failed Paths Analysis
        failed_paths = [path for path in results['detailed_paths'].values() if path['failed']]
        if failed_paths:
            failed_analysis = []
            for path in failed_paths:
                failed_analysis.append({
                    'Simulation_ID': path['simulation_id'],
                    'Failure_Day': path['failure_day'],
                    'Failure_Year': round(path['failure_day'] / 252, 2),
                    'Max_Drawdown_Pct': round(path['max_drawdown_pct'], 2),
                    'Margin_Calls_Count': path['margin_calls_count'],
                    'Total_Interest': round(path['total_interest'], 2)
                })
            
            failed_df = pd.DataFrame(failed_analysis)
            failed_df.to_excel(writer, sheet_name='Failed_Paths_Analysis', index=False)
        
        # Sheet 6: Margin Call Analysis
        margin_call_paths = [path for path in results['detailed_paths'].values() 
                           if path['margin_calls_count'] > 0]
        if margin_call_paths:
            mc_analysis = []
            for path in margin_call_paths:
                mc_analysis.append({
                    'Simulation_ID': path['simulation_id'],
                    'Final_Status': 'Failed' if path['failed'] else 'Survived',
                    'Margin_Calls_Count': path['margin_calls_count'],
                    'Total_Interest': round(path['total_interest'], 2),
                    'Final_Equity': round(path['final_equity'], 2),
                    'Max_Drawdown_Pct': round(path['max_drawdown_pct'], 2),
                    'Interest_Pct_of_Initial': round((path['total_interest'] / 100) * 100, 2)
                })
            
            mc_df = pd.DataFrame(mc_analysis)
            mc_df.to_excel(writer, sheet_name='Margin_Call_Analysis', index=False)
    
    print(f"✅ Excel export completed: {filename}")
    return filename

def run_simulation_and_export():
    """
    Execute complete simulation with Excel export for both leverage levels
    """
    
    print("🚀 Monte Carlo Simulation with Excel Export")
    print("📊 Parameters: 10% return, 10% volatility, 13.7%/12.2% margins\n")
    
    # Common simulation parameters
    common_params = {
        'mu_annual': 0.10,
        'sigma_annual': 0.10,
        'init_margin_rate': 0.137,
        'maint_margin_rate': 0.122,
        'funding_rate_annual': 0.068,
        'years': 5,
        'simulations': 10000,
        'export_sample_size': 1000
    }
    
    # Run 1.6x leverage simulation
    print("1️⃣ Running Leverage 1.6x Simulation...")
    results_16x = futures_portfolio_simulation_with_export(leverage=1.6, **common_params)
    
    # Export 1.6x results
    file_16x = export_simulation_results_to_excel(results_16x, 1.6)
    
    # Run 2.4x leverage simulation  
    print("\n2️⃣ Running Leverage 2.4x Simulation...")
    results_24x = futures_portfolio_simulation_with_export(leverage=2.4, **common_params)
    
    # Export 2.4x results
    file_24x = export_simulation_results_to_excel(results_24x, 2.4)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("📊 Export Summary")
    print(f"{'='*60}")
    
    print(f"\nLeverage 1.6x Results:")
    print(f"   • Failure rate: {results_16x['failures']/10000*100:.3f}%")
    print(f"   • Margin call rate: {results_16x['margin_call_paths']/10000*100:.1f}%")
    print(f"   • Excel file: {file_16x}")
    
    print(f"\nLeverage 2.4x Results:")
    print(f"   • Failure rate: {results_24x['failures']/10000*100:.3f}%")
    print(f"   • Margin call rate: {results_24x['margin_call_paths']/10000*100:.1f}%")
    print(f"   • Excel file: {file_24x}")
    
    print(f"\n📁 Files saved in current directory")
    print(f"📋 Each file contains 6 analysis sheets with 1,000 detailed simulation paths")
    print(f"✅ Ready for detailed Excel analysis!")
    
    return file_16x, file_24x

# Execute the simulation and export
if __name__ == "__main__":
    excel_file_16x, excel_file_24x = run_simulation_and_export()
