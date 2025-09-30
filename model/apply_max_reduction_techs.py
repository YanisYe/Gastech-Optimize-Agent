#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-step level optimization: Select technology combinations by technology level, sorting within each level by level, economic cost, and emission reduction.

Optimization logic:
1. Calculate net emission reduction for each technology using environmental technology impact mechanisms
2. Technology impact formula: g_after = g_before * (1 + delta)
3. Calculation steps:
   a. Calculate total baseline emissions for all counties for each affected sub-industry
   b. Net emission reduction = sum(total baseline emissions * delta) for all affected sub-industries
   c. delta < 0: emission reduction contribution (positive), delta > 0: emission increase contribution (negative)
4. Step-by-step selection strategy:
   a. Step 1: Apply ≤1 level technology package to all counties (sorted by level, economic cost, emission reduction)
   b. Step 2: Reapply ≤2 level technology package to counties that still don't meet standards (sorted by level, economic cost, emission reduction)
   c. Step 3: Reapply ≤3 level technology package to counties that still don't meet standards (sorted by level, economic cost, emission reduction)
5. Conflict handling: For technologies in the same industry and same conflict group, select the optimal one by sorting by level, economic cost, and emission reduction
6. Output the technologies included in each technology package and the technology package allocation used by each county
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import functools
import time
import logging

# Add model directory to path
model_path = Path(__file__).parent / "model"
sys.path.append(str(model_path))

from GasEnviroment_curriculum_learning import GasEnv, GasEnvConfig

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Default output to console
    ]
)
logger = logging.getLogger(__name__)



def get_conflicts_tech(techId, techSet):
    """
    Get the indices of technologies that conflict with the specified technology

    Args:
        techId: Technology ID
        techSet: Technology collection DataFrame

    Returns:
        pandas.Series: Boolean series, True indicates conflicting technology
    """
    if isinstance(techId, torch.Tensor):
        techId = techId.cpu().numpy()
    
    if hasattr(techId, '__len__') and len(techId) > 1:
        techId = techId[0]
    
    target_row = techSet.iloc[techId]
    industry = target_row['class']
    conflict = target_row['技术间的冲突']

    # Handle NaN values: NaN indicates no conflicts
    if pd.isna(conflict):
        # If the target technology has no conflicts, return False series (no conflicting technologies)
        return pd.Series(False, index=techSet.index)

    # Find technologies in the same industry and same conflict group
    condition = (
        (techSet['class'] == industry) &
        (techSet['技术间的冲突'] == conflict)
    )
    
    return condition

def calculate_total_emissions_needing_reduction(env):
    """
    Calculate total emissions of three gases for all counties needing emission reduction (using Total_*_origin data from environment)

    Args:
        env: Environment instance

    Returns:
        dict: Dictionary containing total emissions of NH3, NO3, N_runoff
    """
    # Get indices of counties needing technology
    counties_needing_tech = np.where(env.counties_need_tech)[0]

    # Use total emission data from environment
    total_emissions = {
        'NH3': 0.0,
        'NO3': 0.0,
        'N_runoff': 0.0
    }

    for county_idx in counties_needing_tech:
        # Use Total_*_origin data from environment, these are total emissions for each county
        # Total_NH3_origin = Crop NH3 + Manure management NH3 + Manure application NH3
        # Total_NO3_origin = Fertilizer NO3 + Manure application NO3
        # Total_N_runoff_origin = N runoff
        if hasattr(env, 'Total_NH3_origin') and env.Total_NH3_origin is not None:
            total_emissions['NH3'] += env.Total_NH3_origin[county_idx].item()

        if hasattr(env, 'Total_NO3_origin') and env.Total_NO3_origin is not None:
            total_emissions['NO3'] += env.Total_NO3_origin[county_idx].item()

        if hasattr(env, 'Total_N_runoff_origin') and env.Total_N_runoff_origin is not None:
            total_emissions['N_runoff'] += env.Total_N_runoff_origin[county_idx].item()

    logger.info(f"Total counties needing emission reduction: {len(counties_needing_tech)}")
    logger.info(f"Total emissions - NH3: {total_emissions['NH3']:.4f}, NO3: {total_emissions['NO3']:.4f}, N_runoff: {total_emissions['N_runoff']:.4f}")

    return total_emissions

def _calculate_tech_impacts_base(tech_row, env):
    """
    Base function to calculate the impact of a single technology on all sub-industries
    Return detailed change information for each affected sub-industry

    Args:
        tech_row: Technology data row
        env: Environment instance

    Returns:
        list: List of tuples containing (industry, delta, sub, total_change)
    """
    impacts = []

    try:
        # Get technology ID
        tech_id = tech_row.name  # Assume tech_row index is the technology ID

        # Get technology details
        line = env._get_line(tech_id)
        class_name = 'crop' if 'crop' in line.index[4].lower() else 'livestock'
        line = line[5:]

        # Get technology impact segments (reuse environment methods)
        deltas = env._get_delta(line, class_name)

        # Expand technology sub-industries (reuse environment methods)
        deltas = env.expand_tech_subindustries(deltas, relaxed=False)

        # Iterate through all affected sub-industries
        for industry, delta, subIndustry in deltas:
            # Get current state of technology impact segments
            if industry in env.stateMapping:
                countyState = env.stateMapping[industry]
                subindustry_classes = env.class_mapping[industry]

                # Iterate through sub-segments of technology impact segments
                for sub in subIndustry:
                    # Get index of sub-segment
                    col = env._get_or_create_industry_mapping(sub, subindustry_classes)

                    # Calculate total baseline emissions of this sub-industry across all counties needing emission reduction
                    total_before_value = 0.0
                    for county_idx in np.where(env.counties_need_tech)[0]:
                        try:
                            before_value = float(countyState[county_idx, col].item())
                            total_before_value += before_value
                        except (IndexError, AttributeError):
                            continue

                    # Use correct formula: total change = total baseline emissions * delta
                    if total_before_value > 0:  # Only calculate for sub-industries with emissions
                        total_change = total_before_value * delta
                        impacts.append((industry, delta, sub, total_change))

    except Exception as e:
        logger.warning(f"Failed to calculate base technology impact data: {e}")

    return impacts

def calculate_tech_comprehensive_impacts(tech_row, env):
    """
    Unified calculation of comprehensive impact of a single technology: net emission reduction + various gas and yield impacts

    Args:
        tech_row: Technology data row
        env: Environment instance

    Returns:
        dict: Dictionary containing net emission reduction and various gas yield impacts
    """
    # Initialize result dictionary
    impacts = {
        'net_reduction': 0.0,  # Net emission reduction
        'NH3_reduction': 0.0,
        'NO3_reduction': 0.0,
        'N_runoff_reduction': 0.0,
        'CH4_reduction': 0.0,
        'N2O_reduction': 0.0,
        'SOC_reduction': 0.0,
        'yield_change': 0.0
    }

    try:
        # Use base function to get impacts on all sub-industries
        impacts_base = _calculate_tech_impacts_base(tech_row, env)

        # Classify gas impacts based on sub-industry names
        for industry, delta, sub, total_change in impacts_base:
            sub_lower = sub.lower()

            # Accumulate net emission reduction (excluding certain specific sub-industries)
            if not any(x.lower() in sub_lower for x in ['organic_carbon', 'CH4', 'N2O', 'Yield']):
                impacts['net_reduction'] += total_change

            # Classify and accumulate impacts of various gases and yields
            if 'nh3' in sub_lower:
                impacts['NH3_reduction'] += total_change
            elif 'no3' in sub_lower:
                impacts['NO3_reduction'] += total_change
            elif ('n_runoff' in sub_lower) or ('runoff' in sub_lower and 'n' in sub_lower):
                impacts['N_runoff_reduction'] += total_change
            elif 'ch4' in sub_lower and 'intestine' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'ch4' in sub_lower and 'fecal' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'ch4' in sub_lower and 'rice' in sub_lower:
                impacts['CH4_reduction'] += total_change
            elif 'n2o' in sub_lower:
                impacts['N2O_reduction'] += total_change
            elif 'organic_carbon' in sub_lower or 'soc' in sub_lower:
                impacts['SOC_reduction'] += total_change
            elif 'yield' in sub_lower:
                impacts['yield_change'] += total_change

    except Exception as e:
        logger.warning(f"Failed to calculate comprehensive technology impacts: {e}")

    return impacts

def select_optimal_techs_by_level_and_reduction(env, max_level):
    """
    Select optimal technology combination based on technology level, economic cost, and net emission reduction, considering technology conflicts
    Sorting priority: technology level ascending > economic cost ascending > net emission reduction descending (more negative values have higher priority)
    Use technology impact mechanism in environment: g_after = g_before * (1 + delta)

    Args:
        env: Environment instance
        max_level: Maximum technology level (1, 2, 3)

    Returns:
        tuple: (Selected technology ID list, technology package information dictionary)
    """
    tech_set = env.tech_set

    # Filter technologies within specified level
    available_techs = tech_set[tech_set['技术分级'] <= max_level].copy()
    if len(available_techs) == 0:
        logger.warning(f"No technologies found with level ≤{max_level}")
        return [], {}

    logger.info(f"Starting to select ≤{max_level} level technologies, optimal technology combination...")

    # Calculate total emissions from counties needing emission reduction (for information display)
    total_emissions = calculate_total_emissions_needing_reduction(env)

    # Serially calculate net emission reduction for all available technologies
    logger.info(f"Calculating net emission reduction for ≤{max_level} level technologies ({len(available_techs)})...")
    start_time = time.time()
    tech_reductions = []
    for idx, tech_row in available_techs.iterrows():
        impacts = calculate_tech_comprehensive_impacts(tech_row, env)
        net_reduction = impacts['net_reduction']
        tech_reductions.append((idx, net_reduction, tech_row))
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(".2f")

    # Sort by technology level, economic cost, and net emission reduction: level ascending > cost ascending > reduction ascending
    tech_reductions.sort(key=lambda x: (x[2]['技术分级'], x[2]['经济成本'], x[1]))

    logger.info(f"Calculated net emission reduction for {len(tech_reductions)} ≤{max_level} level technologies")
    logger.info("Top 10 technologies with lowest economic cost:")
    for i, (idx, net_reduction, row) in enumerate(tech_reductions[:10]):
        reduction_type = "Net reduction" if net_reduction <= 0 else "Net increase"
        value = abs(net_reduction)
        logger.info(f"  {i+1}. {row['Mitigation strategy']} (Level:{row['技术分级']}, Cost:{row['经济成本']:.2f}, {reduction_type}:{value:.4f})")

    selected_techs = []
    conflict_groups_used = set()  # Record used conflict groups

    for tech_idx, net_reduction, tech_row in tech_reductions:
        # Check if technology has conflicts
        conflict_value = tech_row['技术间的冲突']

        # If technology has no conflicts (NaN), adopt directly
        if pd.isna(conflict_value):
            selected_techs.append(tech_idx)
            reduction_type = "Net reduction" if net_reduction <= 0 else "Net increase"
            logger.info(f"Selected technology: {tech_row['Mitigation strategy']}, Type: {tech_row['Crop species'] if tech_row['class'] == 'crop' else tech_row['Livestock species']} (Level:{tech_row['技术分级']}, {reduction_type}:{abs(net_reduction):.4f}, Cost:{tech_row['经济成本']:.2f}) - No conflicts")
            continue

        # If there are conflicts, perform conflict detection
        conflict_key = (tech_row['class'], conflict_value)

        if conflict_key in conflict_groups_used:
            # If conflict group is already used, check if current technology is better
            existing_tech_idx = None
            for selected_idx in selected_techs:
                selected_row = available_techs.loc[selected_idx] if selected_idx in available_techs.index else tech_set.iloc[selected_idx]
                selected_conflict = selected_row['技术间的冲突']
                if (selected_row['class'] == tech_row['class'] and
                    selected_conflict == conflict_value):
                    existing_tech_idx = selected_idx
                    break

            if existing_tech_idx is not None:
                existing_row = available_techs.loc[existing_tech_idx] if existing_tech_idx in available_techs.index else tech_set.iloc[existing_tech_idx]
                existing_impacts = calculate_tech_comprehensive_impacts(existing_row, env)
                existing_net_reduction = existing_impacts['net_reduction']

                # Compare: level ascending > economic cost ascending > net emission reduction descending (more negative values have higher priority)
                current_priority = (tech_row['技术分级'], tech_row['经济成本'], net_reduction)
                existing_priority = (existing_row['技术分级'], existing_row['经济成本'], existing_net_reduction)

                if current_priority < existing_priority:
                    # Current technology is better, replace existing technology
                    selected_techs.remove(existing_tech_idx)
                    selected_techs.append(tech_idx)
                    reduction_type_existing = "Net reduction" if existing_net_reduction <= 0 else "Net increase"
                    reduction_type_current = "Net reduction" if net_reduction <= 0 else "Net increase"
                    logger.info(f"Replaced technology: {existing_row['Mitigation strategy']} (Level:{existing_row['技术分级']}, {reduction_type_existing}:{abs(existing_net_reduction):.4f}) -> {tech_row['Mitigation strategy']} (Level:{tech_row['技术分级']}, {reduction_type_current}:{abs(net_reduction):.4f})")
                # else: Keep existing technology
            continue
        else:
            # New conflict group, add directly
            selected_techs.append(tech_idx)
            conflict_groups_used.add(conflict_key)
            reduction_type = "Net reduction" if net_reduction <= 0 else "Net increase"
            logger.info(f"Selected technology: {tech_row['Mitigation strategy']}, Type: {tech_row['Crop species'] if tech_row['class'] == 'crop' else tech_row['Livestock species']} (Level:{tech_row['技术分级']}, {reduction_type}:{abs(net_reduction):.4f}, Cost:{tech_row['经济成本']:.2f})")

    # Count number of technologies at each level
    level_counts = {}
    for tech_idx in selected_techs:
        row = available_techs.loc[tech_idx] if tech_idx in available_techs.index else tech_set.iloc[tech_idx]
        level = row['技术分级']
        level_counts[level] = level_counts.get(level, 0) + 1

    logger.info(f"Finally selected {len(selected_techs)} ≤{max_level} level optimal technologies, level distribution: {level_counts}")

    # Create technology package information
    tech_package_info = {
        'max_level': max_level,
        'selected_techs': selected_techs,
        'level_counts': level_counts,
        'tech_details': []
    }

    for tech_idx in selected_techs:
        row = available_techs.loc[tech_idx] if tech_idx in available_techs.index else tech_set.iloc[tech_idx]

        # Use unified function to calculate all impacts
        comprehensive_impacts = calculate_tech_comprehensive_impacts(row, env)
        net_reduction = comprehensive_impacts['net_reduction']
        reduction_type = "Net reduction" if net_reduction <= 0 else "Net increase"

        tech_package_info['tech_details'].append({
            'tech_idx': tech_idx,
            'name': row['Mitigation strategy'],
            'level': row['技术分级'],
            'net_reduction': net_reduction,
            'reduction_type': reduction_type,
            'cost': row['经济成本'],
            'class': row['class'],
            'species': row['Livestock species'] if row['class'] != 'crop' else row['Crop species'],
            'conflict': '' if pd.isna(row['技术间的冲突']) else row['技术间的冲突'],
            # Add gas and yield impact information
            'NH3_reduction': comprehensive_impacts['NH3_reduction'],
            'NO3_reduction': comprehensive_impacts['NO3_reduction'],
            'N_runoff_reduction': comprehensive_impacts['N_runoff_reduction'],
            'CH4_reduction': comprehensive_impacts['CH4_reduction'],
            'N2O_reduction': comprehensive_impacts['N2O_reduction'],
            'SOC_reduction': comprehensive_impacts['SOC_reduction'],
            'yield_change': comprehensive_impacts['yield_change']
        })

    return selected_techs, tech_package_info

def check_counties_meeting_targets(env):
    """
    Check which counties have met the targets

    Args:
        env: Environment instance

    Returns:
        numpy.ndarray: Boolean array, True indicates the county has met all targets
    """
    # Counties that meet all targets
    all_targets_met = (
        (env.gap_NO3.squeeze() <= 0) & 
        (env.gap_NH3.squeeze() <= 0) & 
        (env.gap_N_runoff.squeeze() <= 0) 
        # (env.gap_CH4.squeeze() <= 0) &
        # (env.gap_N2O.squeeze() <= 0)
    ).cpu().numpy()
    
    return all_targets_met



def apply_techs_to_county_batch(county_batch, env, selected_tech_ids):
    """
    Apply selected technology combination to a batch of counties

    Args:
        county_batch: County batch list [(batch_idx, county_idx), ...]
        env: Environment instance
        selected_tech_ids: Selected technology ID list

    Returns:
        dict: Statistical results for this batch
    """
    batch_actions = 0
    batch_applied_techs = 0
    batch_results = []

    for batch_idx, county_idx in county_batch:
        county_name = env.IDs['Counties'].iloc[county_idx]
        county_applied_techs = 0

        # Apply selected technologies
        for tech_idx in selected_tech_ids:
            # Check if technology has already been applied
            if env.state['Tech_selected'][county_idx, tech_idx] == 0:
                # Encode action: need to convert county_idx to index in counties_need_tech
                all_counties_need_tech = np.where(env.counties_need_tech)[0]
                county_need_tech_pos = np.where(all_counties_need_tech == county_idx)[0][0]
                action = county_need_tech_pos * env.numTech + tech_idx

                try:
                    state, reward, terminated, truncated, info = env.step(action)
                    batch_actions += 1
                    batch_applied_techs += 1
                    county_applied_techs += 1
                except Exception as e:
                    logger.error(f"Error applying technology {tech_idx} to county {county_name}: {e}")
                    continue

        batch_results.append({
            'county_idx': county_idx,
            'county_name': county_name,
            'applied_techs': county_applied_techs,
            'batch_idx': batch_idx
        })

    return {
        'batch_actions': batch_actions,
        'batch_applied_techs': batch_applied_techs,
        'batch_results': batch_results
    }

def apply_techs_to_counties_parallel(env, selected_tech_ids, target_counties=None, num_workers=None):
    """
    Apply selected technology combination to counties using multithreading in parallel

    Args:
        env: Environment instance
        selected_tech_ids: Selected technology ID list
        target_counties: Target county index list, if None then apply to all counties needing technology
        num_workers: Number of worker threads, automatically selected by default

    Returns:
        tuple: (Number of applied technologies, total number of executed actions)
    """
    if not selected_tech_ids:
        logger.warning("No technologies need to be applied, skip")
        return 0, 0

    # Determine target counties
    if target_counties is None:
        counties_need_tech_indices = np.where(env.counties_need_tech)[0]
    else:
        # Only process specified counties, and these counties need to be in counties_need_tech
        counties_need_tech_indices = np.where(env.counties_need_tech)[0]
        target_counties = np.array(target_counties)
        # Take intersection
        counties_need_tech_indices = np.intersect1d(counties_need_tech_indices, target_counties)

    if len(counties_need_tech_indices) == 0:
        logger.warning("No counties need technology, skip")
        return 0, 0

    logger.info(f"Starting parallel application of technologies to {len(counties_need_tech_indices)} counties...")

    # Set number of worker threads
    if num_workers is None:
        num_workers = min(cpu_count(), 24)  # Use at most 24 threads

    # Divide counties into multiple batches
    county_list = list(enumerate(counties_need_tech_indices))
    batch_size = max(1, len(county_list) // num_workers)
    county_batches = [county_list[i:i + batch_size] for i in range(0, len(county_list), batch_size)]

    logger.info(f"Using {num_workers} threads to process {len(county_list)} counties in parallel")
    logger.info(f"Each thread handles approximately {batch_size} counties")

    start_time = time.time()

    # Create a thread-safe counter to track total progress
    total_actions = 0
    total_applied_techs = 0
    completed_batches = 0

    # Use thread pool to process county batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all batch tasks
        future_to_batch = {
            executor.submit(apply_techs_to_county_batch, batch, env, selected_tech_ids): batch_idx
            for batch_idx, batch in enumerate(county_batches)
        }

        # Collect results
        for future in as_completed(future_to_batch):
            try:
                batch_result = future.result()
                total_actions += batch_result['batch_actions']
                total_applied_techs += batch_result['batch_applied_techs']
                completed_batches += 1

                # Display batch completion information
                batch_idx = future_to_batch[future]
                batch_counties = len(batch_result['batch_results'])
                logger.info(f"Completed batch {completed_batches}/{len(county_batches)} ({batch_counties} counties, {batch_result['batch_applied_techs']} technologies)")

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                completed_batches += 1

    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("Parallel technology application completed!")
    logger.info(f"Total applied {total_applied_techs} technologies")
    logger.info(f"Total executed actions: {total_actions}")
    logger.info(".2f")

    return total_applied_techs, total_actions

def apply_tech_package_to_counties(env, target_counties, selected_tech_ids, tech_package_info, num_workers=None, collect_impacts=True):
    """
    Apply technology package to specified counties, and record county technology package allocation and impact of each technology on each county

    Args:
        env: Environment instance
        target_counties: Target county index list
        selected_tech_ids: Technology ID list in technology package
        tech_package_info: Technology package information dictionary
        num_workers: Number of parallel technology application threads, automatically selected by default
        collect_impacts: Whether to collect technology impact data, default True

    Returns:
        tuple: (Number of applied technologies, total number of executed actions, county technology package allocation dictionary, technology impact data dictionary)
    """
    logger.info(f"Starting application of ≤{tech_package_info['max_level']} level technology package to {len(target_counties)} counties...")

    # Initialize county technology package assignment records
    county_package_assignment = {}

    # Initialize technology impact data records
    tech_county_impacts = {
        'step_level': tech_package_info['max_level'],
        'applied_techs': selected_tech_ids,
        'county_impacts': {}  # county_idx -> {tech_idx -> impact_data}
    } if collect_impacts else None

    if not selected_tech_ids:
        logger.warning("Technology package is empty, skip application")
        return 0, 0, county_package_assignment, tech_county_impacts

    # Determine target counties (only process specified counties)
    target_counties = np.array(target_counties)

    logger.info(f"Number of target counties: {len(target_counties)}")

    # Initialize variables
    total_actions = 0
    applied_techs = 0


    # Use serial method
    logger.info("Using serial method to apply technologies...")
    total_actions = 0
    applied_techs = 0

    for i, county_idx in enumerate(target_counties):
        county_name = env.IDs['Counties'].iloc[county_idx]
        logger.info(f"Processing county {i+1}/{len(target_counties)}: {county_name}")

        # Record which technology package this county uses
        county_package_assignment[county_idx] = tech_package_info['max_level']

        # Initialize impact data for this county
        if collect_impacts and county_idx not in tech_county_impacts['county_impacts']:
            tech_county_impacts['county_impacts'][county_idx] = {
                'county_name': county_name,
                'applied_techs': [],
                'total_impacts': {
                    'NH3_change': 0.0,
                    'NO3_change': 0.0,
                    'N_runoff_change': 0.0,
                    'CH4_change': 0.0,
                    'N2O_change': 0.0,
                    'SOC_change': 0.0,
                    'yield_change': 0.0,
                    'net_reduction': 0.0
                }
            }

        # If impact data needs to be collected, record initial state before applying technology
        initial_state = {}
        if collect_impacts and hasattr(env, 'state') and env.state:
            for key, value in env.state.items():
                if key != 'Tech_selected':
                    # For 2D states, extract the value for this county
                    initial_state[key] = float(value[county_idx].item())
            initial_state['yield'] = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item())

        # Apply selected technologies
        county_applied_techs = 0
        for tech_idx in selected_tech_ids:
            # Check if technology has already been applied
            if env.state['Tech_selected'][county_idx, tech_idx] == 0:
                action = county_idx * env.numTech + tech_idx


                state, reward, terminated, truncated, info = env.step(action)
                total_actions += 1
                applied_techs += 1
                county_applied_techs += 1

                # If impact data needs to be collected, extract impact from returned state
                if collect_impacts:
                    tech_row = env.tech_set.iloc[tech_idx]
                    tech_name = tech_row['Mitigation strategy']

                    # Extract gas indicators from state after technology application
                    impact_data = {
                        'county_name': county_name,
                        'NH3_change': 0.0,
                        'NO3_change': 0.0,
                        'N_runoff_change': 0.0,
                        'CH4_change': 0.0,
                        'N2O_change': 0.0,
                        'SOC_change': 0.0,
                        'yield_change': 0.0,
                        'net_reduction': 0.0
                    }

                    # Calculate state changes after technology application
                    for key, value in state.items():
                        if key == 'Tech_selected':
                            continue
                        # Classify gas impacts based on state key
                        key_lower = key.lower()
                        # For 2D states, extract current value for this county
                        current_value = float(value[county_idx].item())

                        # Calculate change amount (current value minus initial value)
                        initial_value = initial_state.get(key, 0.0)
                        change = current_value - initial_value
                        # Only record states with changes, avoid accumulating 0 values
                        if change != 0:
                            if 'nh3' in key_lower:
                                impact_data['NH3_change'] += change
                                impact_data['net_reduction'] += change
                            
                            elif 'no3' in key_lower:
                                impact_data['NO3_change'] += change
                                impact_data['net_reduction'] += change
                                
                            elif ('n_runoff' in key_lower):
                                impact_data['N_runoff_change'] += change
                                impact_data['net_reduction'] += change

                            elif 'ch4' in key_lower:
                                impact_data['CH4_change'] += change
                            elif 'n2o' in key_lower:
                                impact_data['N2O_change'] += change
                            elif 'organic' in key_lower or 'soc' in key_lower:
                                impact_data['SOC_change'] += change

                    tech_line = env._get_line(tech_idx)
                    if 'Yield' in tech_line:
                        change = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item()) - initial_state['yield']
                        impact_data['yield_change'] += change

                    # Save technology impact data
                    tech_county_impacts['county_impacts'][county_idx]['applied_techs'].append({
                        'tech_idx': tech_idx,
                        'tech_name': tech_name,
                        'impacts': impact_data
                    })

                    # for key, value in impact_data.items():
                    #     if value != 0:
                    #         logger.info(f"{key}: {value}")

                    # Accumulate to county's total impacts
                    for key in tech_county_impacts['county_impacts'][county_idx]['total_impacts']:
                        tech_county_impacts['county_impacts'][county_idx]['total_impacts'][key] += impact_data[key]

                    # Update initial state to reflect current state after technology application
                    # This way, when applying technology next time, impact will be calculated based on the final state of the previous technology
                    for key, value in state.items():
                        if key != 'Tech_selected':
                            # Update the initial state of this county to the state after technology application
                            initial_state[key] = float(value[county_idx].item())
                    initial_state['yield'] = float(env.stateMapping['畜牧业产量'][county_idx].sum().item()) + float(env.stateMapping['种植业产量'][county_idx].sum().item())
        
        logger.info(f"County {county_name} actually applied {county_applied_techs} technologies")

        # Print progress every 10 counties processed
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i+1} counties, total applied {applied_techs} technologies")

        logger.info("Serial technology application completed!")
        logger.info(f"Total applied {applied_techs} technologies")
        logger.info(f"Total executed actions: {total_actions}")

    return applied_techs, total_actions, county_package_assignment, tech_county_impacts

def save_tech_package_to_excel(tech_package, step_name, output_dir="results", suffix="level_based_stepwise_techs"):
    """
    Immediately save individual technology package information to Excel file

    Args:
        tech_package: Technology package information dictionary
        step_name: Step name (e.g. 'step_1')
        output_dir: Output directory
        suffix: File name suffix
    """
    if not tech_package or 'tech_details' not in tech_package:
        return

    # Create output directory
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # Prepare technology package data
    tech_package_data = []
    for tech_detail in tech_package['tech_details']:
        tech_package_data.append({
            '步骤': step_name.upper(),
            '最大等级': tech_package.get('max_level', 0),
            '技术ID': tech_detail['tech_idx'],
            '技术名称': tech_detail['name'],
            '技术等级': tech_detail['level'],
            '净减排量': tech_detail['net_reduction'],
            '减排类型': tech_detail['reduction_type'],
            '经济成本': tech_detail['cost'],
            '产业类型': tech_detail['class'],
            '物种': tech_detail.get('species', ''),
            '冲突组': '' if pd.isna(tech_detail['conflict']) else tech_detail['conflict'],
            # Add gas and yield impact information
            'NH3减排量': tech_detail.get('NH3_reduction', 0.0),
            'NO3减排量': tech_detail.get('NO3_reduction', 0.0),
            'N_runoff减排量': tech_detail.get('N_runoff_reduction', 0.0),
            'CH4减排量': tech_detail.get('CH4_reduction', 0.0),
            'N2O减排量': tech_detail.get('N2O_reduction', 0.0),
            'SOC减排量': tech_detail.get('SOC_reduction', 0.0),
            '产量变化': tech_detail.get('yield_change', 0.0)
        })

    # Save to Excel
    if tech_package_data:
        tech_package_df = pd.DataFrame(tech_package_data)
        tech_package_file = os.path.join(full_output_dir, f"{step_name}_tech_package.xlsx")
        tech_package_df.to_excel(tech_package_file, index=False)
        logger.info(f"{step_name.upper()} technology package information saved to: {tech_package_file}")

def stepwise_level_based_tech_optimization(env_config, use_parallel=False, num_workers=None):
    """
    Step-by-step level optimization: Optimize by technology level in steps, first level 1, then 1+2, finally 1+2+3
    Within each level, sort by level, economic cost, emission reduction, consider technology conflicts, apply appropriate technology packages based on compliance status

    Args:
        env: Environment instance
        env_config: Environment configuration object, used to create new environment instances
        use_parallel: Whether to use parallel technology application, default False
        num_workers: Number of parallel technology application threads, automatically selected by default

    Returns:
        dict: Optimization statistics and county technology package allocation information
    """
    logger.info("=" * 60)
    logger.info("Starting step-by-step level optimization (step by technology level + sort by level, economic cost, emission reduction)")
    logger.info("=" * 60)

    stats = {
        'step_1': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 1, 'tech_package': {}},
        'step_2': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 2, 'tech_package': {}},
        'step_3': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 3, 'tech_package': {}}
    }

    # Record technology package allocation for all counties
    all_county_assignments = {}

    # Step 1: Apply ≤1 level technology package to all counties needing technology
    logger.info(f"\n" + "="*50)
    env_step1 = GasEnv(env_config)
    env_step1.reset()
    logger.info("Step 1: Apply ≤1 level technology package to all counties needing emission reduction")
    logger.info("="*50)

    total_counties = env_step1.numCounty
    counties_need_tech_indices = np.where(env_step1.counties_need_tech)[0]
    logger.info(f"Total counties: {total_counties}, counties needing emission reduction: {len(counties_need_tech_indices)}")

    if len(counties_need_tech_indices) > 0:
        selected_tech_ids_1, tech_package_1 = select_optimal_techs_by_level_and_reduction(env_step1, max_level=1)
        stats['step_1']['tech_package'] = tech_package_1

        if selected_tech_ids_1:
            applied_techs_1, actions_1, assignments_1, _ = apply_tech_package_to_counties(
                env_step1, counties_need_tech_indices, selected_tech_ids_1, tech_package_1,
                num_workers=num_workers, collect_impacts=False
            )
            stats['step_1']['applied_techs'] = applied_techs_1
            stats['step_1']['actions'] = actions_1
            all_county_assignments.update(assignments_1)
        else:
            logger.warning("No level 1 technologies found")

    # Check compliance status after step 1
    counties_met_after_step1 = check_counties_meeting_targets(env_step1)
    counties_met_count_1 = counties_met_after_step1.sum()
    stats['step_1']['counties_met'] = counties_met_count_1

    logger.info(f"Compliance status after step 1 application:")
    logger.info(f"Compliant counties: {counties_met_count_1}/{total_counties} ({counties_met_count_1/total_counties*100:.1f}%)")

    # Save technology package information immediately after step 1
    if stats['step_1']['tech_package']:
        save_tech_package_to_excel(stats['step_1']['tech_package'], 'step_1')
        logger.info("Step 1 technology package saved successfully")

    # Step 2: Apply ≤2 level technology package to counties that don't meet standards
    counties_not_met_after_step1 = np.where(~counties_met_after_step1)[0]
    if len(counties_not_met_after_step1) > 0:
        logger.info(f"\n" + "="*50)
        logger.info(f"Step 2: Reapply ≤2 level technology package to {len(counties_not_met_after_step1)} counties that don't meet standards")
        logger.info("="*50)

        # Create new environment instance for step 2 to ensure clean state
        logger.info("Creating new environment instance for step 2...")
        env_step2 = GasEnv(env_config)
        env_step2.reset()
        logger.info("Step 2 environment instance created successfully")

        selected_tech_ids_2, tech_package_2 = select_optimal_techs_by_level_and_reduction(env_step2, max_level=2)
        stats['step_2']['tech_package'] = tech_package_2

        if selected_tech_ids_2:
            applied_techs_2, actions_2, assignments_2, _ = apply_tech_package_to_counties(
                env_step2, counties_not_met_after_step1, selected_tech_ids_2, tech_package_2,
                num_workers=num_workers, collect_impacts=False
            )
            stats['step_2']['applied_techs'] = applied_techs_2
            stats['step_2']['actions'] = actions_2
            all_county_assignments.update(assignments_2)

        # Check compliance status after step 2
        counties_met_after_step2 = check_counties_meeting_targets(env_step2)
        counties_met_count_2 = counties_met_after_step2.sum()
        stats['step_2']['counties_met'] = counties_met_count_2

        # Calculate union of counties that meet requirements for level 1 technology package and 1-2 level technology package
        # That is, all counties that meet standards after step 1 or step 2
        counties_met_union = np.logical_or(counties_met_after_step1, counties_met_after_step2)
        counties_met_union_count = counties_met_union.sum()

        logger.info(f"Union of counties meeting requirements for level 1 technology package and 1-2 level technology packages:")
        logger.info(f"Union compliant counties: {counties_met_union_count}/{total_counties} ({counties_met_union_count/total_counties*100:.1f}%)")
        logger.info(f"Step 1 compliant counties: {counties_met_count_1}")
        logger.info(f"Step 2 compliant counties: {counties_met_count_2}")
        logger.info(f"Union additional counties: {counties_met_union_count - counties_met_count_1}")

        # Also save union results to statistics
        stats['step_2']['counties_met_union'] = counties_met_union_count

        logger.info(f"Compliance status after step 2 application:")
        logger.info(f"Compliant counties: {counties_met_count_2}/{total_counties} ({counties_met_count_2/total_counties*100:.1f}%)")
        logger.info(f"New compliant counties added: {counties_met_count_2 - counties_met_count_1}")

        # Save technology package information immediately after step 2
        if stats['step_2']['tech_package']:
            save_tech_package_to_excel(stats['step_2']['tech_package'], 'step_2')
            logger.info("Step 2 technology package saved successfully")

        # Step 3: Apply ≤3 level technology package to counties not in union that don't meet targets
        # Calculate counties not in union (i.e., counties that still don't meet standards)
        counties_not_in_union = np.where(~counties_met_union)[0]
        if len(counties_not_in_union) > 0:
            logger.info(f"\n" + "="*50)
            logger.info(f"Step 3: Reapply ≤3 level technology package to {len(counties_not_in_union)} counties not in union that don't meet targets")
            logger.info(f"Union compliant counties: {counties_met_union_count}, counties still needing step 3: {len(counties_not_in_union)}")
            logger.info("="*50)

            # Create new environment instance for step 3 to ensure clean state
            logger.info("Creating new environment instance for step 3...")
            env_step3 = GasEnv(env_config)
            env_step3.reset()
            logger.info("Step 3 environment instance created successfully")

            selected_tech_ids_3, tech_package_3 = select_optimal_techs_by_level_and_reduction(env_step3, max_level=3)
            stats['step_3']['tech_package'] = tech_package_3

            if selected_tech_ids_3:
                applied_techs_3, actions_3, assignments_3, _ = apply_tech_package_to_counties(
                    env_step3, counties_not_in_union, selected_tech_ids_3, tech_package_3,
                    num_workers=num_workers, collect_impacts=False
                )
                stats['step_3']['applied_techs'] = applied_techs_3
                stats['step_3']['actions'] = actions_3
                all_county_assignments.update(assignments_3)

            # Check compliance status after step 3
            counties_met_after_step3 = check_counties_meeting_targets(env_step3)
            counties_met_count_3 = counties_met_after_step3.sum()
            stats['step_3']['counties_met'] = counties_met_count_3

            # Calculate union of three technology packages (level 1, 1-2 level, 1-3 level technology packages)
            # Union of three technology packages = Level 1 compliant ∪ 1-2 level compliant ∪ 1-3 level compliant
            counties_met_three_tech_packages = np.logical_or(
                np.logical_or(counties_met_after_step1, counties_met_after_step2),
                counties_met_after_step3
            )
            counties_met_three_tech_packages_count = counties_met_three_tech_packages.sum()

            logger.info(f"Compliance status after step 3 application:")
            logger.info(f"Compliant counties: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"New compliant counties added in step 3: {counties_met_three_tech_packages_count - counties_met_union_count}")

            logger.info(f"\nUnion details of three technology packages:")
            logger.info(f"Counties meeting level 1 technology package requirements: {counties_met_count_1}")
            logger.info(f"Counties meeting 1-2 level technology package requirements: {counties_met_union_count}")
            logger.info(f"Counties meeting 1-3 level technology package requirements: {counties_met_three_tech_packages_count}")
            logger.info(f"Union additional counties compared to step 1: {counties_met_three_tech_packages_count - counties_met_count_1}")
            logger.info(f"Union additional counties compared to step 2: {counties_met_three_tech_packages_count - counties_met_union_count}")

            # Also save union results of three technology packages to statistics
            stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count

            # Save technology package information immediately after step 3
            if stats['step_3']['tech_package']:
                save_tech_package_to_excel(stats['step_3']['tech_package'], 'step_3')
                logger.info("Step 3 technology package saved successfully")
        else:
            logger.info(f"All counties are in the union of counties that meet requirements for level 1 technology package and 1-2 level technology package, no need for step 3")

            # Calculate the union of three technology packages (at this point step 3 was not executed, so union equals level 1 and 1-2 union)
            counties_met_three_tech_packages_count = counties_met_union_count

            logger.info(f"\nCompliance status after step 3 application:")
            logger.info(f"Compliant counties: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"Total new compliant counties added: {counties_met_three_tech_packages_count - counties_met_count_1}")

            logger.info(f"\nUnion details of three technology packages:")
            logger.info(f"Counties meeting level 1 technology package requirements: {counties_met_count_1}")
            logger.info(f"Counties meeting 1-2 level technology package requirements: {counties_met_union_count}")
            logger.info(f"Counties meeting 1-3 level technology package requirements: Step 3 not executed, no additional")
            logger.info(f"Union compliant counties of three technology packages: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")
            logger.info(f"Union additional counties compared to step 1: {counties_met_three_tech_packages_count - counties_met_count_1}")

            # Also save union results of three technology packages to statistics
            stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count
    else:
        logger.info(f"All counties met standards after step 1, no need for subsequent steps")

        # Calculate the union of three technology packages (at this point steps 2 and 3 were not executed, so union equals level 1 compliance)
        counties_met_three_tech_packages_count = counties_met_count_1

        logger.info(f"\nCompliance status after step 3 application:")
        logger.info(f"Compliant counties: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")

        logger.info(f"\nUnion details of three technology packages:")
        logger.info(f"Counties meeting level 1 technology package requirements: {counties_met_count_1}")
        logger.info(f"Counties meeting 1-2 level technology package requirements: Step 2 not executed, equals step 1")
        logger.info(f"Counties meeting 1-3 level technology package requirements: Step 3 not executed, equals step 1")
        logger.info(f"Union compliant counties of three technology packages: {counties_met_three_tech_packages_count}/{total_counties} ({counties_met_three_tech_packages_count/total_counties*100:.1f}%)")

        # Also save union results of three technology packages to statistics
        stats['step_3']['counties_met_three_tech_packages'] = counties_met_three_tech_packages_count

    # Output overall statistics
    logger.info(f"\n" + "=" * 60)
    logger.info("Step-by-step level optimization completed! Overall statistics:")
    logger.info("=" * 60)

    total_applied_techs = stats['step_1']['applied_techs'] + stats['step_2']['applied_techs'] + stats['step_3']['applied_techs']
    total_actions = stats['step_1']['actions'] + stats['step_2']['actions'] + stats['step_3']['actions']

    # Use the union count of counties meeting targets from three technology packages as final compliant county count
    final_met_counties = stats['step_3']['counties_met_three_tech_packages']

    logger.info(f"Step 1 (≤Level 1 tech): Applied {stats['step_1']['applied_techs']} technologies, {stats['step_1']['actions']} actions")
    logger.info(f"Step 2 (≤Level 2 tech): Applied {stats['step_2']['applied_techs']} technologies, {stats['step_2']['actions']} actions")
    logger.info(f"Step 3 (≤Level 3 tech): Applied {stats['step_3']['applied_techs']} technologies, {stats['step_3']['actions']} actions")
    logger.info(f"Total: Applied {total_applied_techs} technologies, {total_actions} actions")
    logger.info(f"Final compliant counties: {final_met_counties}/{total_counties} ({final_met_counties/total_counties*100:.1f}%)")

    # Add county technology package allocation information to statistics results
    stats['county_assignments'] = all_county_assignments

    return stats

def load_tech_packages_from_files(output_dir="results/level_based_stepwise_techs"):
    """
    Load technology package information from Excel files

    Args:
        output_dir: Directory where technology package files are located

    Returns:
        dict: Dictionary containing technology package information and county allocation information
    """
    logger.info("Loading technology package information from files...")

    # Initialize statistics structure
    stats = {
        'step_1': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 1, 'tech_package': {}},
        'step_2': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 2, 'tech_package': {}},
        'step_3': {'applied_techs': 0, 'actions': 0, 'counties_met': 0, 'max_level': 3, 'tech_package': {}}
    }

    # Load technology package information
    tech_package_files = [
        os.path.join(output_dir, "step_1_tech_package.xlsx"),
        os.path.join(output_dir, "step_2_tech_package.xlsx"),
        os.path.join(output_dir, "step_3_tech_package.xlsx")
    ]

    # Load county assignment information
    county_assignment_file = os.path.join(output_dir, "level_based_stepwise_techs", "county_tech_assignments.xlsx")
    if os.path.exists(county_assignment_file):
        try:
            county_df = pd.read_excel(county_assignment_file)
            # Build county_assignments dictionary
            county_assignments = {}
            for _, row in county_df.iterrows():
                county_assignments[row['县索引']] = row['分配技术包等级']
            stats['county_assignments'] = county_assignments
            logger.info(f"County allocation information loaded, total {len(county_assignments)} counties")
        except Exception as e:
            logger.error(f"Failed to load county allocation file: {e}")
            stats['county_assignments'] = {}
    else:
        logger.warning(f"County allocation file not found: {county_assignment_file}")
        stats['county_assignments'] = {}

    # Load technology packages for each step
    for i, file_path in enumerate(tech_package_files, 1):
        if os.path.exists(file_path):
            try:
                tech_df = pd.read_excel(file_path)

                # Build technology package information
                tech_package = {
                    'max_level': i,
                    'selected_techs': tech_df['技术ID'].tolist() if '技术ID' in tech_df.columns else [],
                    'tech_details': []
                }

                # Build technology details, ensure field names match original format
                for _, row in tech_df.iterrows():
                    tech_detail = {
                        'tech_idx': row.get('技术ID', row.get('tech_idx', 0)),
                        'name': row.get('技术名称', row.get('name', 'Unknown')),
                        'level': row.get('技术等级', row.get('level', 1)),
                        'net_reduction': row.get('净减排量', row.get('net_reduction', 0.0)),
                        'reduction_type': row.get('减排类型', row.get('reduction_type', '净减排')),
                        'cost': row.get('经济成本', row.get('cost', 0.0)),
                        'class': row.get('产业类型', row.get('class', 'unknown')),
                        'species': row.get('物种', row.get('species', '')),
                        'conflict': row.get('冲突组', row.get('conflict', '')),
                        # Add gas and yield impact information
                        'NH3_reduction': row.get('NH3减排量', row.get('NH3_reduction', 0.0)),
                        'NO3_reduction': row.get('NO3减排量', row.get('NO3_reduction', 0.0)),
                        'N_runoff_reduction': row.get('N_runoff减排量', row.get('N_runoff_reduction', 0.0)),
                        'CH4_reduction': row.get('CH4减排量', row.get('CH4_reduction', 0.0)),
                        'N2O_reduction': row.get('N2O减排量', row.get('N2O_reduction', 0.0)),
                        'SOC_reduction': row.get('SOC减排量', row.get('SOC_reduction', 0.0)),
                        'yield_change': row.get('产量变化', row.get('yield_change', 0.0))
                    }
                    tech_package['tech_details'].append(tech_detail)

                # Calculate technology count and action count (approximate values)
                tech_count = len(tech_package['selected_techs'])
                # Assume each county applies all technologies in the package on average
                county_count = len([idx for idx, level in stats['county_assignments'].items() if level == i])
                actions_count = tech_count * county_count

                stats[f'step_{i}']['tech_package'] = tech_package
                stats[f'step_{i}']['applied_techs'] = tech_count
                stats[f'step_{i}']['actions'] = actions_count

                logger.info(f"Step {i} technology package loaded: {tech_count} technologies, expected {actions_count} actions")

            except Exception as e:
                logger.error(f"Error loading technology package file {file_path}: {e}")
                stats[f'step_{i}']['tech_package'] = {}
        else:
            logger.warning(f"Technology package file not found: {file_path}")
            stats[f'step_{i}']['tech_package'] = {}

    # Calculate overall statistics
    total_applied_techs = sum(stats[f'step_{i}']['applied_techs'] for i in [1, 2, 3])
    total_actions = sum(stats[f'step_{i}']['actions'] for i in [1, 2, 3])

    # Calculate compliant county count (use sum of counties assigned to each level technology package as approximation)
    final_met_counties = len(stats['county_assignments'])
    total_counties = final_met_counties  # Cannot accurately get total county count here, use assigned county count as approximation

    logger.info(f"Technology package loading completed:")
    logger.info(f"Total: Applied {total_applied_techs} technologies, {total_actions} actions")
    logger.info(f"Counties involved: {final_met_counties}")

    # Set final compliant county count
    stats['step_3']['counties_met_three_tech_packages'] = final_met_counties

    return stats

def save_tech_county_impacts(env, output_dir="results", suffix="level_based_stepwise_techs", tech_impacts_data=None):
    """
    Save impact data of gas indicators for each technology on each county

    Args:
        env: Environment instance
        output_dir: Output directory
        suffix: File name suffix
    """
    logger.info("Generating impact data for each technology on each county...")

    # Create output directory
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # Get all technologies
    tech_set = env.tech_set
    county_names = env.IDs['Counties'].tolist()

    # Create impact data files for each technology
    tech_county_impacts_dir = os.path.join(full_output_dir, "tech_county_impacts")
    os.makedirs(tech_county_impacts_dir, exist_ok=True)

    # If actual technology impact data is provided, use it; otherwise recalculate
    if tech_impacts_data:
        logger.info("Using impact data from actual technology applications...")
        # Extract impacts of all technologies from technology impact data
        all_tech_impacts = {}
        for step_key, step_data in tech_impacts_data.items():
            if step_key.startswith('step_') and 'tech_impacts' in step_data:
                step_impacts = step_data['tech_impacts']
                for county_idx, county_data in step_impacts['county_impacts'].items():
                    for tech_data in county_data['applied_techs']:
                        tech_idx = tech_data['tech_idx']
                        if tech_idx not in all_tech_impacts:
                            all_tech_impacts[tech_idx] = {}
                        all_tech_impacts[tech_idx][county_idx] = tech_data['impacts']

        # Create impact data files for each technology
        for tech_idx, county_impacts in all_tech_impacts.items():
            try:
                tech_row = tech_set.iloc[tech_idx]
                tech_name = tech_row['Mitigation strategy']

                # Prepare data for saving - only process counties that need technology
                impact_data = []

                # Get indices of counties that need technology
                counties_need_tech_indices = np.where(env.counties_need_tech)[0]

                # Only iterate through counties that need technology
                for county_idx in counties_need_tech_indices:
                    if county_idx in county_impacts:
                        impact_data.append({
                            '县名称': county_names[county_idx],
                            '县索引': county_idx,
                            'NH3变化': county_impacts[county_idx]['NH3_change'],
                            'NO3变化': county_impacts[county_idx]['NO3_change'],
                            'N_runoff变化': county_impacts[county_idx]['N_runoff_change'],
                            'CH4变化': county_impacts[county_idx]['CH4_change'],
                            'N2O变化': county_impacts[county_idx]['N2O_change'],
                            'SOC变化': county_impacts[county_idx]['SOC_change'],
                            '产量变化': county_impacts[county_idx]['yield_change'],
                            '净减排量': county_impacts[county_idx]['net_reduction']
                        })
                    else:
                        # For counties that need technology but did not apply this technology, show 0 values
                        impact_data.append({
                            '县名称': county_names[county_idx],
                            '县索引': county_idx,
                            'NH3变化': 0.0,
                            'NO3变化': 0.0,
                            'N_runoff变化': 0.0,
                            'CH4变化': 0.0,
                            'N2O变化': 0.0,
                            'SOC变化': 0.0,
                            '产量变化': 0.0,
                            '净减排量': 0.0
                        })

                # Save to Excel file
                safe_tech_name = "".join(c for c in tech_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_tech_name = safe_tech_name.replace(' ', '_').replace('-', '_')

                impact_df = pd.DataFrame(impact_data)
                impact_file = os.path.join(tech_county_impacts_dir, f"tech_{tech_idx}_{safe_tech_name[:50]}.xlsx")
                impact_df.to_excel(impact_file, index=False)

            except Exception as e:
                logger.error(f"Error processing technology {tech_idx}: {e}")
                continue
    else:
        # If technology impact data is not provided, log warning but do not perform calculation
        logger.warning("Technical impact data not provided, skipping individual technology impact file generation")

    logger.info(f"Technology impact data on counties saved to directory: {tech_county_impacts_dir}")

    # Create summary file showing which technologies each county applied and their impacts
    try:
        logger.info("Generating county technology impact summary...")

        # If actual technology impact data is provided, use it; otherwise calculate from environment state
        if tech_impacts_data:
            # Use actual technology impact data
            applied_techs = {}
            county_summary_data = []

            # Extract county technology application information from technology impact data
            for step_key, step_data in tech_impacts_data.items():
                if 'tech_impacts' in step_data:
                    step_impacts = step_data['tech_impacts']
                    for county_idx, county_data in step_impacts['county_impacts'].items():
                        if county_idx not in applied_techs:
                            applied_techs[county_idx] = []

                        # Record technologies applied to this county
                        for tech_data in county_data['applied_techs']:
                            tech_idx = tech_data['tech_idx']
                            if tech_idx not in applied_techs[county_idx]:
                                applied_techs[county_idx].append(tech_idx)

            # Create summary data for each county - only process counties that need technology
            # Get indices of counties that need technology
            counties_need_tech_indices = np.where(env.counties_need_tech)[0]

            for county_idx in counties_need_tech_indices:
                county_name = county_names[county_idx]
                applied_tech_list = applied_techs.get(county_idx, [])

                # Calculate the total impact of all technologies applied to this county
                total_impacts = {
                    'NH3变化': 0.0,
                    'NO3变化': 0.0,
                    'N_runoff变化': 0.0,
                    'CH4变化': 0.0,
                    'N2O变化': 0.0,
                    'SOC变化': 0.0,
                    '产量变化': 0.0,
                    '净减排量': 0.0
                }

                tech_details = []
                for step_key, step_data in tech_impacts_data.items():
                    if 'tech_impacts' in step_data:
                        step_impacts = step_data['tech_impacts']
                        if county_idx in step_impacts['county_impacts']:
                            county_data = step_impacts['county_impacts'][county_idx]

                            for tech_data in county_data['applied_techs']:
                                tech_idx = tech_data['tech_idx']
                                tech_row = tech_set.iloc[tech_idx]
                                tech_name = tech_row['Mitigation strategy']
                                impact = tech_data['impacts']

                                tech_details.append({
                                    '技术ID': tech_idx,
                                    '技术名称': tech_name,
                                    'NH3变化': impact['NH3_change'],
                                    'NO3变化': impact['NO3_change'],
                                    'N_runoff变化': impact['N_runoff_change'],
                                    'CH4变化': impact['CH4_change'],
                                    'N2O变化': impact['N2O_change'],
                                    'SOC变化': impact['SOC_change'],
                                    '产量变化': impact['yield_change'],
                                    '净减排量': impact['net_reduction']
                                })

                                # Accumulate to county's total impacts
                                # Create key name mapping
                                key_mapping = {
                                    'NH3变化': 'NH3_change',
                                    'NO3变化': 'NO3_change',
                                    'N_runoff变化': 'N_runoff_change',
                                    'CH4变化': 'CH4_change',
                                    'N2O变化': 'N2O_change',
                                    'SOC变化': 'SOC_change',
                                    '产量变化': 'yield_change',
                                    '净减排量': 'net_reduction'
                                }

                                for key in total_impacts:
                                    if key in key_mapping:
                                        total_impacts[key] += impact[key_mapping[key]]

                county_summary_data.append({
                    '县名称': county_name,
                    '县索引': county_idx,
                    '应用技术数量': len(applied_tech_list),
                    '技术ID列表': ','.join(map(str, applied_tech_list)),
                    '总NH3变化': total_impacts['NH3变化'],
                    '总NO3变化': total_impacts['NO3变化'],
                    '总N_runoff变化': total_impacts['N_runoff变化'],
                    '总CH4变化': total_impacts['CH4变化'],
                    '总N2O变化': total_impacts['N2O变化'],
                    '总SOC变化': total_impacts['SOC变化'],
                    '总产量变化': total_impacts['产量变化'],
                    '总净减排量': total_impacts['净减排量']
                })
        else:
            # If technology impact data is not provided, create empty summary data
            logger.warning("Technical impact data not provided, unable to generate county summary")
            county_summary_data = []

        # Save county summary data
        county_summary_df = pd.DataFrame(county_summary_data)
        county_summary_file = os.path.join(full_output_dir, "county_tech_impacts_summary.xlsx")
        county_summary_df.to_excel(county_summary_file, index=False)
        logger.info(f"County technology impact summary saved to: {county_summary_file}")

        # Save detailed technology application data (impact of each technology on each county)
        detailed_data = []
        if tech_impacts_data:
            # Use actual technology impact data
            for step_key, step_data in tech_impacts_data.items():
                if 'tech_impacts' in step_data:
                    step_impacts = step_data['tech_impacts']
                    for county_idx, county_data in step_impacts['county_impacts'].items():
                        county_name = county_names[county_idx]

                        for tech_data in county_data['applied_techs']:
                            tech_idx = tech_data['tech_idx']
                            tech_row = tech_set.iloc[tech_idx]
                            impact = tech_data['impacts']

                            detailed_data.append({
                                '县名称': county_name,
                                '县索引': county_idx,
                                '技术ID': tech_idx,
                                '技术名称': tech_row['Mitigation strategy'],
                                '技术等级': tech_row['技术分级'],
                                '产业类型': tech_row['class'],
                                'NH3变化': impact['NH3_change'],
                                'NO3变化': impact['NO3_change'],
                                'N_runoff变化': impact['N_runoff_change'],
                                'CH4变化': impact['CH4_change'],
                                'N2O变化': impact['N2O_change'],
                                'SOC变化': impact['SOC_change'],
                                '产量变化': impact['yield_change'],
                                '净减排量': impact['net_reduction']
                            })
        else:
            # If technology impact data is not provided, skip detailed data generation
            logger.warning("Technical impact data not provided, skipping detailed data generation")

        if detailed_data:
            detailed_df = pd.DataFrame(detailed_data)
            detailed_file = os.path.join(full_output_dir, "county_tech_impacts_detailed.xlsx")
            detailed_df.to_excel(detailed_file, index=False)
            logger.info(f"County technology impact details saved to: {detailed_file}")

    except Exception as e:
        logger.error(f"Error generating county technology impact summary: {e}")

def output_state_summary(env, output_dir="results", suffix="level_based_stepwise_techs", optimization_stats=None, skip_tech_save=False):
    """
    Output environment state summary, including technology package information and county technology package allocation

    Args:
        env: Environment instance
        output_dir: Output directory
        suffix: File name suffix
        optimization_stats: Optimization statistics information, including technology package and county allocation information
    """
    logger.info(f"\n=== Output state after step-by-step level optimization ===")

    # Create output directory
    full_output_dir = os.path.join(output_dir, suffix)
    os.makedirs(full_output_dir, exist_ok=True)

    # Output and save technology package information
    if optimization_stats:
        logger.info(f"\n=== Technology Package Information ===")

        # Prepare technology package data for saving
        tech_package_data = []

        for step_key, step_data in optimization_stats.items():
            if step_key.startswith('step_') and 'tech_package' in step_data:
                tech_package = step_data['tech_package']
                if tech_package and 'tech_details' in tech_package:
                    logger.info(f"\n{step_key.upper()} Technology Package (≤{tech_package.get('max_level', '?')} level):")
                    logger.info(f"  Technology count: {len(tech_package['tech_details'])}")
                    logger.info(f"  Level distribution: {tech_package.get('level_counts', {})}")

                    for tech_detail in tech_package['tech_details']:
                        logger.info(f"    - {tech_detail['name']} (Level:{tech_detail['level']}, {tech_detail['reduction_type']}:{abs(tech_detail['net_reduction']):.4f}, Cost:{tech_detail['cost']:.2f})")
                        logger.info(f"      Gas impacts - NH3:{tech_detail.get('NH3_reduction', 0):.4f}, NO3:{tech_detail.get('NO3_reduction', 0):.4f}, N_runoff:{tech_detail.get('N_runoff_reduction', 0):.4f}")
                        logger.info(f"      Gas impacts - CH4:{tech_detail.get('CH4_reduction', 0):.4f}, N2O:{tech_detail.get('N2O_reduction', 0):.4f}, SOC:{tech_detail.get('SOC_reduction', 0):.4f}, Yield:{tech_detail.get('yield_change', 0):.4f}")

                        # Add to technology package data list
                        tech_package_data.append({
                            '步骤': step_key.upper(),
                            '最大等级': tech_package.get('max_level', 0),
                            '技术ID': tech_detail['tech_idx'],
                            '技术名称': tech_detail['name'],
                            '技术等级': tech_detail['level'],
                            '净减排量': tech_detail['net_reduction'],
                            '减排类型': tech_detail['reduction_type'],
                            '经济成本': tech_detail['cost'],
                            '产业类型': tech_detail['class'],
                            '物种': tech_detail.get('species', ''),
                            '冲突组': '' if pd.isna(tech_detail['conflict']) else tech_detail['conflict']
                        })

            # Save technology package information to Excel
        if not skip_tech_save and tech_package_data:
            tech_package_df = pd.DataFrame(tech_package_data)
            tech_package_file = os.path.join(full_output_dir, "tech_packages.xlsx")
            tech_package_df.to_excel(tech_package_file, index=False)
            logger.info(f"Technology package information saved to: {tech_package_file}")
        elif skip_tech_save:
            logger.info("Skip technology package information saving (file already exists)")

        # Output county technology package allocation information
        if 'county_assignments' in optimization_stats:
            logger.info(f"\n=== County Technology Package Allocation ===")
            county_names = env.IDs['Counties'].tolist()
            assignments = optimization_stats['county_assignments']

            level_1_counties = []
            level_2_counties = []
            level_3_counties = []
            no_tech_counties = []

            for county_idx in range(len(county_names)):
                if county_idx in assignments:
                    level = assignments[county_idx]
                    if level == 1:
                        level_1_counties.append(county_names[county_idx])
                    elif level == 2:
                        level_2_counties.append(county_names[county_idx])
                    elif level == 3:
                        level_3_counties.append(county_names[county_idx])
                else:
                    no_tech_counties.append(county_names[county_idx])

            logger.info(f"Counties using level 1 technology package ({len(level_1_counties)}): {', '.join(level_1_counties[:10])}{'...' if len(level_1_counties) > 10 else ''}")
            logger.info(f"Counties using level 2 technology package ({len(level_2_counties)}): {', '.join(level_2_counties[:10])}{'...' if len(level_2_counties) > 10 else ''}")
            logger.info(f"Counties using level 3 technology package ({len(level_3_counties)}): {', '.join(level_3_counties[:10])}{'...' if len(level_3_counties) > 10 else ''}")
            logger.info(f"Counties requiring no technology intervention ({len(no_tech_counties)}): {', '.join(no_tech_counties[:10])}{'...' if len(no_tech_counties) > 10 else ''}")

            # Save county technology package allocation information to Excel
            county_assignment_data = []
            for county_idx in range(len(county_names)):
                county_name = county_names[county_idx]
                if county_idx in assignments:
                    package_level = assignments[county_idx]
                    county_assignment_data.append({
                        '县名称': county_name,
                        '县索引': county_idx,
                        '分配技术包等级': package_level,
                        '技术包类型': f'≤{package_level}级技术包'
                    })
                else:
                    county_assignment_data.append({
                        '县名称': county_name,
                        '县索引': county_idx,
                        '分配技术包等级': 0,
                        '技术包类型': '无需技术干预'
                    })

            if not skip_tech_save:
                county_assignment_df = pd.DataFrame(county_assignment_data)
                county_assignment_file = os.path.join(full_output_dir, "county_tech_assignments.xlsx")
                county_assignment_df.to_excel(county_assignment_file, index=False)
                logger.info(f"County technology package allocation information saved to: {county_assignment_file}")
            else:
                logger.info("Skip county technology package allocation information saving (file already exists)")
    
    # Get county name list
    county_names = env.IDs['Counties'].tolist()
    
    # 1. Save observation state data (env.state)
    state_data = {}
    for key, value in env.state.items():
        if key == 'Tech_selected':
            # Calculate the number of technologies selected by each county
            tech_counts = value[:, :env.numTech].sum(axis=1).cpu().numpy()
            state_data['选择技术数量'] = tech_counts
        elif isinstance(value, torch.Tensor):
            if value.dim() == 2 and value.shape[1] == 1:
                state_data[key] = value.squeeze().cpu().numpy()
            elif value.dim() == 1:
                state_data[key] = value.cpu().numpy()
    
    # Create observation state DataFrame
    state_df = pd.DataFrame(state_data, index=county_names)
    state_file = os.path.join(full_output_dir, "state_after_single_step_techs.xlsx")
    state_df.to_excel(state_file)
    logger.info(f"Observation state data saved to: {state_file}")

    # 2. Save all state data from stateMapping (save by key separately)
    if hasattr(env, 'stateMapping') and env.stateMapping:
        logger.info(f"Saving {len(env.stateMapping)} state mappings...")
        for key, value in env.stateMapping.items():
            try:
                if isinstance(value, torch.Tensor):
                    # Get corresponding column names
                    if key in env.class_mapping:
                        columns = env.class_mapping[key]
                        if hasattr(columns, 'tolist'):
                            columns = columns.tolist()
                        elif hasattr(columns, '__iter__') and not isinstance(columns, str):
                            columns = list(columns)
                        else:
                            columns = [str(columns)]
                    else:
                        # If no mapping, use default column names
                        columns = [f"Col_{i}" for i in range(value.shape[1])] if value.dim() > 1 else ["Value"]

                    # Create DataFrame
                    if value.dim() == 1:
                        value_2d = value.unsqueeze(1)
                        columns = ["Value"]
                    else:
                        value_2d = value

                    state_mapping_df = pd.DataFrame(
                        value_2d.detach().cpu().numpy(),
                        index=county_names,
                        columns=columns
                    )
                else:
                    # Handle non-tensor data
                    if key in env.class_mapping:
                        columns = env.class_mapping[key]
                        if hasattr(columns, 'tolist'):
                            columns = columns.tolist()
                        elif hasattr(columns, '__iter__') and not isinstance(columns, str):
                            columns = list(columns)
                        else:
                            columns = [str(columns)]
                    else:
                        columns = None

                    state_mapping_df = pd.DataFrame(value, index=county_names, columns=columns)

                # Save file
                state_mapping_file = os.path.join(full_output_dir, f"{key}.xlsx")
                state_mapping_df.to_excel(state_mapping_file)
                logger.info(f"State mapping '{key}' saved to: {state_mapping_file}")

            except Exception as e:
                logger.error(f"Save state mapping '{key}' failed: {e}")
                continue
    
    # 3. Emission reduction gap data
    gap_data = {
        'gap_NO3': env.gap_NO3.squeeze().cpu().numpy(),
        'gap_N_runoff': env.gap_N_runoff.squeeze().cpu().numpy(),
        'gap_NH3': env.gap_NH3.squeeze().cpu().numpy(),
        'gap_CH4': env.gap_CH4.squeeze().cpu().numpy(),
        'gap_N2O': env.gap_N2O.squeeze().cpu().numpy()
    }
    
    gap_df = pd.DataFrame(gap_data, index=county_names)
    gap_file = os.path.join(full_output_dir, "gaps_after_single_step_techs.xlsx")
    gap_df.to_excel(gap_file)
    logger.info(f"Emission reduction gap data saved to: {gap_file}")

    # 4. Statistical summary
    logger.info("\n=== Statistical summary of each indicator ===")
    logger.info("Observation state indicator statistics:")
    logger.info(f"\n{state_df.describe()}")

    logger.info("\nEmission reduction gap statistics:")
    logger.info(f"\n{gap_df.describe()}")

    # If stateMapping is saved, output list of state mapping files
    if hasattr(env, 'stateMapping') and env.stateMapping:
        logger.info(f"\nSaved state mapping files:")
        for key in env.stateMapping.keys():
            logger.info(f"  - {key}.xlsx")
    
    # 5. Compliance status statistics
    no3_met = (env.gap_NO3.squeeze() <= 0).sum().item()
    nh3_met = (env.gap_NH3.squeeze() <= 0).sum().item()
    n_runoff_met = (env.gap_N_runoff.squeeze() <= 0).sum().item()
    ch4_met = (env.gap_CH4.squeeze() <= 0).sum().item()
    n2o_met = (env.gap_N2O.squeeze() <= 0).sum().item()
    
    total_counties = len(county_names)
    
    logger.info(f"\n=== Compliance status statistics (Total counties: {total_counties}) ===")
    logger.info(f"NO3 met: {no3_met} counties ({no3_met/total_counties*100:.1f}%)")
    logger.info(f"NH3 met: {nh3_met} counties ({nh3_met/total_counties*100:.1f}%)")
    logger.info(f"N_runoff met: {n_runoff_met} counties ({n_runoff_met/total_counties*100:.1f}%)")
    logger.info(f"CH4 met: {ch4_met} counties ({ch4_met/total_counties*100:.1f}%)")
    logger.info(f"N2O met: {n2o_met} counties ({n2o_met/total_counties*100:.1f}%)")
    
    # Number of counties meeting all indicators
    all_targets_met = (
        (env.gap_NO3.squeeze() <= 0) & 
        (env.gap_NH3.squeeze() <= 0) & 
        (env.gap_N_runoff.squeeze() <= 0) 
        # (env.gap_CH4.squeeze() <= 0) &
        # (env.gap_N2O.squeeze() <= 0)
    ).sum().item()
    
    logger.info(f"All indicators met: {all_targets_met} counties ({all_targets_met/total_counties*100:.1f}%)")

    # 6. Save statistical summary
    summary_data = {
        '指标': ['总县数', 'NO3达标县数', 'NH3达标县数', 'N_runoff达标县数', 'CH4达标县数', 'N2O达标县数', '全部达标县数'],
        '数量': [total_counties, no3_met, nh3_met, n_runoff_met, ch4_met, n2o_met, all_targets_met],
        '比例(%)': [100.0, no3_met/total_counties*100, nh3_met/total_counties*100,
                   n_runoff_met/total_counties*100, ch4_met/total_counties*100,
                   n2o_met/total_counties*100, all_targets_met/total_counties*100]
    }

    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(full_output_dir, "compliance_statistics_summary.xlsx")
    summary_df.to_excel(summary_file, index=False)
    logger.info(f"Compliance statistics summary saved to: {summary_file}")

    # 7. Save cost data
    if hasattr(env, 'save_path') and env.save_path is not None:
        # Ensure save path exists
        if not os.path.exists(env.save_path):
            os.makedirs(env.save_path, exist_ok=True)
        cost_df = env._save_tech_selected_summary()
        # Also save cost data to output_dir
        cost_df.to_excel(os.path.join(full_output_dir, "cost.xlsx"))
        logger.info(f"Cost data saved to: {os.path.join(full_output_dir, 'cost.xlsx')}")
    else:
        logger.warning("Environment save_path not set, skipping cost data saving")

    # Save impact data of each technology on each county
    try:
        # If there is technology impact data, pass it to save function
        tech_impacts_data = None
        if optimization_stats:
            tech_impacts_data = {}
            for step_key, step_data in optimization_stats.items():
                if step_key.startswith('step_') and 'tech_impacts' in step_data:
                    tech_impacts_data[step_key] = {'tech_impacts': step_data['tech_impacts']}

        save_tech_county_impacts(env, output_dir, suffix, tech_impacts_data)
    except Exception as e:
        logger.error(f"Error saving technology impacts on each county: {e}")

    return state_df, gap_df, summary_df, cost_df if 'cost_df' in locals() else None

def create_tech_impact_summary(output_dir="results/level_based_stepwise_techs"):
    """
    Create summary table of technology gas impact
    Accumulate and summarize impacts of all technologies on all counties
    """
    import pandas as pd
    import os
    import glob
    from pathlib import Path

    logger.info("Creating technology impact summary table...")

    # Build tech_county_impacts directory path
    tech_impacts_dir = os.path.join(output_dir, "level_based_stepwise_techs", "tech_county_impacts")

    if not os.path.exists(tech_impacts_dir):
        logger.error(f"Technology impact directory does not exist: {tech_impacts_dir}")
        return

    # Get all technology files
    tech_files = glob.glob(os.path.join(tech_impacts_dir, "tech_*.xlsx"))
    logger.info(f"Found {len(tech_files)} technology files")

    if not tech_files:
        logger.error("No technology impact files found")
        return

    # Initialize summary data
    tech_summary = {}

    # Process each technology file
    for i, file_path in enumerate(tech_files):
        try:
            # Extract technology ID and name from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('.xlsx', '').split('_', 2)
            if len(parts) >= 3:
                tech_id = int(parts[1])
                tech_name = parts[2]
            else:
                logger.warning(f"Cannot parse filename: {filename}")
                continue

            # Read technology impact data
            df = pd.read_excel(file_path)

            # Calculate the total impact of this technology on all counties
            total_impacts = {
                'NH3变化': df['NH3变化'].sum(),
                'NO3变化': df['NO3变化'].sum(),
                'N_runoff变化': df['N_runoff变化'].sum(),
                'CH4变化': df['CH4变化'].sum(),
                'N2O变化': df['N2O变化'].sum(),
                'SOC变化': df['SOC变化'].sum(),
                '产量变化': df['产量变化'].sum(),
                '净减排量': df['净减排量'].sum(),
                '影响县数': len(df[df[['NH3变化', 'NO3变化', 'N_runoff变化', 'CH4变化', 'N2O变化', 'SOC变化', '产量变化']].sum(axis=1) != 0])
            }

            tech_summary[tech_id] = {
                'tech_name': tech_name,
                'tech_id': tech_id,
                **total_impacts
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue

        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(tech_files)} technology files")

    # Create summary DataFrame
    summary_data = []
    for tech_id, data in tech_summary.items():
        summary_data.append(data)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)

        # Sort by net emission reduction (largest absolute values first)
        summary_df['净减排量绝对值'] = summary_df['净减排量'].abs()
        summary_df = summary_df.sort_values('净减排量绝对值', ascending=False)
        summary_df = summary_df.drop('净减排量绝对值', axis=1)

        # Save summary table
        output_file = os.path.join(output_dir, "level_based_stepwise_techs", "tech_gas_impact_summary.xlsx")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        summary_df.to_excel(output_file, index=False)

        logger.info(f"Technology impact summary table saved to: {output_file}")
        logger.info(f"Summarized {len(summary_data)} technology impact data")

        # Output statistics information
        logger.info("Summary statistics:")
        logger.info(f"- Technology total: {len(summary_data)}")
        logger.info(f"- Technology with impact: {len(summary_df[summary_df['净减排量'] != 0])}")
        logger.info(f"- Average impact county number: {summary_df['影响县数'].mean():.1f}")
        logger.info(f"- Maximum net emission reduction: {summary_df['净减排量'].max():.4f}")
        logger.info(f"- Minimum net emission reduction: {summary_df['净减排量'].min():.4f}")

    else:
        logger.error("No valid technology impact data")

def main():
    """
    Main function
    """
    # Configure log file output
    log_file = "results/level_based_stepwise_techs/optimization.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Add file handler to global logger
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("Step-by-step level technology optimization: Step by technology level (Level 1→1+2→1+2+3), sort within each level by level, economic cost, emission reduction")
    logger.info("=" * 60)

    # Check if three technology packages have been saved, if so skip execution
    output_dir = "results/level_based_stepwise_techs"
    tech_package_files = [
        os.path.join(output_dir, "step_1_tech_package.xlsx"),
        os.path.join(output_dir, "step_2_tech_package.xlsx"),
        os.path.join(output_dir, "step_3_tech_package.xlsx")
    ]

    def check_tech_package_file_valid(file_path):
        """检查技术包文件是否存在且包含有效数据"""
        if not os.path.exists(file_path):
            return False
        try:
            df = pd.read_excel(file_path)
            return len(df) > 0  # Ensure file contains data
        except Exception as e:
            logger.warning(f"Error reading technology package file {file_path}: {e}")
            return False

    all_files_valid = all(check_tech_package_file_valid(file) for file in tech_package_files)

    skip_tech_selection = False
    if all_files_valid:
        logger.info("Detected three valid technology package files, skipping technology package selection process")
        logger.info("Existing technology package files:")
        for file in tech_package_files:
            df = pd.read_excel(file)
            logger.info(f"  - {file} (contains {len(df)} technologies)")
        logger.info("Proceeding with existing technology packages")
        logger.info("=" * 60)
        skip_tech_selection = True

    try:
        # Create environment configuration - using Songnen-Sanjiang Plain agricultural area as example
        config = GasEnvConfig(
            Reward_priority=[0.7, 0.5, 0.3, 0.2],
            county_df_path='data/基础数据-县级尺度.xlsx',
            IDs_df='data/县市亚区.xlsx',
            livestock_tech_path='data/畜牧业技术列单-经济产量0827.xlsx',
            crop_tech_path="data/种植业技术列单产量产业0803.xlsx",
            soc_df="data/SOC-县尺度.xlsx",
            livestock_scale="data/动物数量.xlsx",
            crop_scale="data/分县种植面积.xlsx",
            linear_result_path='results/linear_optimization_results_by_county_5gases_hard_target.xlsx',
            only_lp_phase=True,  # Enable linear programming constraint mode
            save_path="temp_yield_output"  # Temporary path for enabling yield tracking
        )
        logger.info("Environment configuration created successfully")

        # Create environment
        env = GasEnv(config)
        # Save configuration for creating new environment instances later
        env_config = config
        logger.info(f"Environment created successfully, containing {env.numCounty} counties, {env.numTech} technologies")

        # Reset environment
        env.reset()
        logger.info("Environment reset to initial state")

        # Output initial state information
        logger.info(f"Number of counties needing technology addition: {env.counties_need_tech.sum()}")
        
        # Execute step-by-step level technology optimization (step by level + sort by level, economic cost, emission reduction)
        if skip_tech_selection:
            logger.info("Skipping technology package selection, loading technology package information from files...")
            optimization_stats = load_tech_packages_from_files("results/level_based_stepwise_techs")
        else:
            optimization_stats = stepwise_level_based_tech_optimization(env_config, use_parallel=False)

        # Reapply all selected technologies in the original environment to ensure final state is correct
        logger.info("Reapplying all selected technologies in the original environment...")

        # Determine which counties each step should apply to based on county_assignments
        if 'county_assignments' in optimization_stats and optimization_stats['county_assignments']:
            counties_step1 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 1]
            counties_step2 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 2]
            counties_step3 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 3]

            logger.info(f"Step 1 technologies will be applied to {len(counties_step1)} counties")
            logger.info(f"Step 2 technologies will be applied to {len(counties_step2)} counties")
            logger.info(f"Step 3 technologies will be applied to {len(counties_step3)} counties")
        else:
            logger.warning("Valid county technology package allocation information not found, will rerun optimization process to generate allocation information")
            logger.info("Starting to rerun step-by-step level optimization...")
            optimization_stats = stepwise_level_based_tech_optimization(env_config, use_parallel=False)

            if 'county_assignments' in optimization_stats and optimization_stats['county_assignments']:
                counties_step1 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 1]
                counties_step2 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 2]
                counties_step3 = [idx for idx, level in optimization_stats['county_assignments'].items() if level == 3]

                logger.info(f"After recalculation, step 1 technologies will be applied to {len(counties_step1)} counties")
                logger.info(f"After recalculation, step 2 technologies will be applied to {len(counties_step2)} counties")
                logger.info(f"After recalculation, step 3 technologies will be applied to {len(counties_step3)} counties")
            else:
                logger.error("County technology package allocation information still not found after rerunning optimization process")
                return

        # Create new environment instance
        env = GasEnv(env_config)
        env.reset()
        logger.info("Environment reset to initial state")

        # Step 1: Apply to counties assigned to step 1
        if counties_step1 and optimization_stats['step_1']['tech_package'] and optimization_stats['step_1']['tech_package']['selected_techs']:
            logger.info(f"Applying Step 1 technology package to {len(counties_step1)} counties...")
            _, _, _, tech_impacts_1 = apply_tech_package_to_counties(env, counties_step1,
                                         sorted(optimization_stats['step_1']['tech_package']['selected_techs']),
                                         optimization_stats['step_1']['tech_package'], collect_impacts=True)
            logger.info("Step 1 technology application completed")

        # Step 2: Apply to counties assigned to step 2
        if counties_step2 and optimization_stats['step_2']['tech_package'] and optimization_stats['step_2']['tech_package']['selected_techs']:
            logger.info(f"Applying Step 2 technology package to {len(counties_step2)} counties...")
            _, _, _, tech_impacts_2 = apply_tech_package_to_counties(env, counties_step2,
                                         sorted(optimization_stats['step_2']['tech_package']['selected_techs']),
                                         optimization_stats['step_2']['tech_package'], collect_impacts=True)
            logger.info("Step 2 technology application completed")

        # Step 3: Apply to counties assigned to step 3
        if counties_step3 and optimization_stats['step_3']['tech_package'] and optimization_stats['step_3']['tech_package']['selected_techs']:
            logger.info(f"Applying Step 3 technology package to {len(counties_step3)} counties...")
            _, _, _, tech_impacts_3 = apply_tech_package_to_counties(env, counties_step3,
                                         sorted(optimization_stats['step_3']['tech_package']['selected_techs']),
                                         optimization_stats['step_3']['tech_package'], collect_impacts=True)
            logger.info("Step 3 technology application completed")
        else:
            logger.warning("County technology package allocation information not found, skipping technology reapplication")

        logger.info("Technology reapplication completed")

        # Collect impact data from all steps
        final_tech_impacts = {}
        if 'tech_impacts_1' in locals() and tech_impacts_1:
            final_tech_impacts['step_1'] = {'tech_impacts': tech_impacts_1}
        if 'tech_impacts_2' in locals() and tech_impacts_2:
            final_tech_impacts['step_2'] = {'tech_impacts': tech_impacts_2}
        if 'tech_impacts_3' in locals() and tech_impacts_3:
            final_tech_impacts['step_3'] = {'tech_impacts': tech_impacts_3}

        # Add collected impact data to optimization statistics
        for step_key, impacts_data in final_tech_impacts.items():
            if step_key in optimization_stats:
                optimization_stats[step_key]['tech_impacts'] = impacts_data['tech_impacts']

        # Output final state
        # If optimization process was rerun, do not skip technology saving
        final_skip_tech_save = skip_tech_selection
        if not ('county_assignments' in optimization_stats or optimization_stats['county_assignments']):
            final_skip_tech_save = False  # Optimization process was rerun, need to save files

        state_df, gap_df, summary_df, cost_df = output_state_summary(env, output_dir="results/level_based_stepwise_techs", suffix="level_based_stepwise_techs", optimization_stats=optimization_stats, skip_tech_save=final_skip_tech_save)

        # Create technology impact summary table
        try:
            create_tech_impact_summary(output_dir="results/level_based_stepwise_techs")
        except Exception as e:
            logger.error(f"Error creating technology impact summary table: {e}")

        # Save optimization statistics data
        if not skip_tech_selection or not ('county_assignments' in optimization_stats and optimization_stats['county_assignments']):
            stats_data = []
            for step, stats in optimization_stats.items():
                if step.startswith('step_'):
                    step_num = step.split('_')[1]
                    stats_data.append({
                        '优化步骤': f'第{step_num}步（≤{stats["max_level"]}级技术包）',
                        '应用技术数量': stats['applied_techs'],
                        '执行动作数': stats['actions'],
                        '达标县数': stats['counties_met']
                    })

            stats_df = pd.DataFrame(stats_data)
            stats_file = os.path.join("results/level_based_stepwise_techs", "optimization_stats.xlsx")
            stats_df.to_excel(stats_file, index=False)
            logger.info(f"Optimization statistics saved to: {stats_file}")
        else:
            logger.info("Skipping optimization statistics save (file already exists)")

        # Clean up temporary directory
        import shutil
        if os.path.exists("temp_yield_output"):
            shutil.rmtree("temp_yield_output")
            logger.info("Temporary yield output directory cleaned up")

        logger.info("\n" + "=" * 60)
        if skip_tech_selection:
            logger.info("Re-optimization based on existing technology packages completed! Results updated to 'results/level_based_stepwise_techs' directory")
        else:
            logger.info("Step-by-step level technology optimization completed! All results saved to 'results/level_based_stepwise_techs' directory")

        logger.info("Included files:")
        if not skip_tech_selection:
            logger.info("  - step_1_tech_package.xlsx: Step 1 technology package details")
            logger.info("  - step_2_tech_package.xlsx: Step 2 technology package details")
            logger.info("  - step_3_tech_package.xlsx: Step 3 technology package details")
            logger.info("  - tech_packages.xlsx: All technology packages summary information")
            logger.info("  - county_tech_assignments.xlsx: County technology package assignments")
            logger.info("  - optimization_stats.xlsx: Step-by-step level optimization statistics")

        logger.info("  - state_after_single_step_techs.xlsx: Observation state data")
        logger.info("  - gaps_after_single_step_techs.xlsx: Emission reduction gap data")
        logger.info("  - compliance_statistics_summary.xlsx: Compliance statistics summary")
        logger.info("  - cost.xlsx: Cost data")
        logger.info("  - county_tech_impacts_summary.xlsx: County technology impacts summary")
        logger.info("  - county_tech_impacts_detailed.xlsx: County technology impacts detailed")
        logger.info("  - tech_gas_impact_summary.xlsx: Technology gas impact summary table")
        logger.info("  - tech_county_impacts/ directory: Detailed impact data for each technology")
        logger.info("  - Various state mapping files (saved by key separately)")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        # Only run summary function
        create_tech_impact_summary()
    else:
        main()