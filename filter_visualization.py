import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use("TkAgg")

def visualize_solution(optimizer, solution=None, save_path=None, ax=None, fig=None):
    """
    Visualize the filter placement solution with unused areas highlighted in red.
    
    Parameters:
    -----------
    optimizer : FilterPlacementOptimizer
        The optimizer instance containing area parameters
    solution : list, optional
        The solution to visualize. If None, uses the best solution found.
    save_path : str, optional
        Path to save the visualization. If None, displays it interactively.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, creates a new figure and axes.
    fig : matplotlib.figure.Figure, optional
        The figure to plot on. If None, creates a new figure.
    scale_factor : float, optional
        If the solution was scaled, provide the scale factor to adjust display settings
    """
    if optimizer.area_length > 1000 or optimizer.area_width > 1000:
        area_dims = f"{optimizer.area_length:,.2f} x {optimizer.area_width:,.2f}"
    else:
        area_dims = f"{optimizer.area_length:.2f} x {optimizer.area_width:.2f}"

    if solution is None:
        solution = optimizer.best_solution
        
    if not solution:
        print("No solution to visualize")
        return
        
    # Update the gap mask for this solution
    optimizer.update_gap_mask(solution)
        
    # Create figure and axis if not provided
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        ax.clear()
    
    # Draw the total area
    ax.add_patch(patches.Rectangle((0, 0), optimizer.area_length, optimizer.area_width, 
                                  fill=True, color='lightgray', alpha=0.3))
    
    # Draw the effective area (after edge gap)
    ax.add_patch(patches.Rectangle((optimizer.edge_gap, optimizer.edge_gap), 
                                  optimizer.effective_length, optimizer.effective_width, 
                                  fill=False, linestyle='--', edgecolor='blue', linewidth=2))
    
    # Count filter types used in this solution
    used_filter_types = set()
    for _, length, width, _, _ in solution:
        # Ensure consistent orientation for counting (longer side first)
        if length < width:
            used_filter_types.add((width, length))
        else:
            used_filter_types.add((length, width))
    
    # Convert to list and sort by area (larger first)
    used_filter_types = sorted(list(used_filter_types), key=lambda size: size[0] * size[1], reverse=True)
    
    # Map filter types to letters (A, B, C, etc.)
    filter_type_to_letter = {size: chr(65 + i) for i, size in enumerate(used_filter_types)}
    
    # Generate colors for filter types
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    colors = distinct_colors[:len(used_filter_types)]
    color_map = {size: colors[i] for i, size in enumerate(used_filter_types)}
    
    # Count filters by size
    filter_counts = {}
    
    for filter_id, length, width, start_x, start_y in solution:
        # Adjust coordinates to include edge gap
        x = start_x + optimizer.edge_gap
        y = start_y + optimizer.edge_gap
        
        # Get filter type (ensure consistent orientation)
        filter_size = (length, width) if (length, width) in used_filter_types else (width, length)
        
        # Get letter and color for this filter type
        letter = filter_type_to_letter[filter_size]
        color = color_map[filter_size]
        
        # Update filter counts
        if filter_size not in filter_counts:
            filter_counts[filter_size] = 0
        filter_counts[filter_size] += 1
        
        # Draw the filter
        ax.add_patch(patches.Rectangle((x, y), length, width, 
                                      fill=True, color=color, alpha=0.7, 
                                      linestyle='-', edgecolor='black'))
        
        # Add filter letter label
        ax.text(x + length/2, y + width/2, letter, 
                ha='center', va='center', fontsize=10, fontweight='bold', color='black')
    
    # Highlight truly unused areas in red (not gaps)
    # For large areas, we need to be more selective about displaying unused points
    area_size = optimizer.effective_length * optimizer.effective_width
    is_large_area = area_size > 100000
    
    if is_large_area:
        # For large areas, we'll sample the unused spaces rather than plotting them all
        # This prevents excessive plotting that would slow down rendering
        sample_rate = max(1, int(optimizer.grid_length * optimizer.grid_width / 5000))
        print(f"Using sampling rate of 1/{sample_rate} for displaying unused areas")
        
        unused_points = []
        for gx in range(0, optimizer.grid_length, sample_rate):
            for gy in range(0, optimizer.grid_width, sample_rate):
                if gy < optimizer.gap_mask.shape[0] and gx < optimizer.gap_mask.shape[1] and optimizer.gap_mask[gy, gx] == 0:
                    # Only highlight truly unused space
                    x = gx * optimizer.grid_resolution + optimizer.edge_gap
                    y = gy * optimizer.grid_resolution + optimizer.edge_gap
                    unused_points.append((x, y))
        
        # Add unused area as a collection rather than individual points for efficiency
        if unused_points:
            unused_x, unused_y = zip(*unused_points)
            ax.scatter(
                [x + optimizer.grid_resolution/2 for x in unused_x],
                [y + optimizer.grid_resolution/2 for y in unused_y],
                color='red', s=20, alpha=0.6, marker='.'
            )
    else:
        # Original approach for smaller areas
        dot_size = 10  # Size of dots representing unused space
        for gx in range(optimizer.grid_length):
            for gy in range(optimizer.grid_width):
                if gy < optimizer.gap_mask.shape[0] and gx < optimizer.gap_mask.shape[1] and optimizer.gap_mask[gy, gx] == 0:
                    # Only highlight truly unused space (not filters or gaps)
                    # Convert back to actual coordinates
                    x = gx * optimizer.grid_resolution + optimizer.edge_gap
                    y = gy * optimizer.grid_resolution + optimizer.edge_gap
                    
                    # Draw a small red dot
                    ax.plot(x + optimizer.grid_resolution/2, y + optimizer.grid_resolution/2, 'ro', 
                           markersize=dot_size/2, alpha=0.6)
                       
    # Set plot limits
    ax.set_xlim(0, optimizer.area_length)
    ax.set_ylim(0, optimizer.area_width)
    
    # Set equal aspect ratio for accurate visual representation
    ax.set_aspect('equal')
    
    # Add title and labels
    coverage, filter_count, filter_variety = optimizer._calculate_metrics(solution)
    
    # Calculate percentages of the effective area
    filter_cells = np.sum(optimizer.gap_mask == 1)
    gap_cells = np.sum(optimizer.gap_mask == 2)
    unused_cells = np.sum(optimizer.gap_mask == 0)
    total_cells = optimizer.grid_width * optimizer.grid_length

    filter_percent = filter_cells / total_cells * 100
    gap_percent = gap_cells / total_cells * 100
    unused_percent = unused_cells / total_cells * 100

    # Create title with info about selected types - now with 2 decimal places
    if hasattr(optimizer, 'selected_filter_types') and optimizer.selected_filter_types:
        selected_types_text = f"Selected {len(optimizer.selected_filter_types)} filter types from {len(optimizer.filter_sizes)} options"
        selected_types_detail = ", ".join([f"{l}x{w}" for l, w in optimizer.selected_filter_types])
        title = (f"Filter Placement\n"
                f"Coverage: {coverage*100:.2f}% of available area\n"
                f"Filters: {filter_count}, Types: {filter_variety}\n"
                f"{selected_types_text}: {selected_types_detail}\n"
                f"Area breakdown: Filters {filter_percent:.2f}%, Required Gaps {gap_percent:.2f}%, Unused {unused_percent:.2f}%")
    else:
        title = (f"Filter Placement\n"
                f"Coverage: {coverage*100:.2f}% of available area\n"
                f"Filters: {filter_count}, Types: {filter_variety}\n"
                f"Area breakdown: Filters {filter_percent:.2f}%, Required Gaps {gap_percent:.2f}%, Unused {unused_percent:.2f}%")
    
    ax.set_title(title)
    ax.set_xlabel("Length")
    ax.set_ylabel("Width")
    
    # Add legend for filter sizes - place it outside the plot area
    legend_patches = []
    for size, count in filter_counts.items():
        letter = filter_type_to_letter[size]
        label = f"Type {letter}: {size[0]}x{size[1]} ({count})"
        legend_patches.append(patches.Patch(color=color_map[size], label=label))
    
    # Add unused space to legend
    legend_patches.append(patches.Patch(color='red', label='Unused Space (Not Gaps)', alpha=0.6))
    
    # Create legend outside the plot area with smaller font
    ax.legend(handles=legend_patches, 
             loc='center left', 
             bbox_to_anchor=(1.02, 0.5),
             fontsize=8,
             handlelength=1.5,
             handleheight=1.5,
             title="Filter Types",
             title_fontsize=9)
    
    # Add annotations for gaps as text in a frame rather than annotations
    gap_info = f"Edge Gap: {optimizer.edge_gap}\nFilter Gap: {optimizer.filter_gap}"
    ax.text(0.02, 0.98, gap_info, 
           transform=ax.transAxes,
           va='top', ha='left',
           fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust figure size to maintain aspect ratio while giving enough space for labels and legend
    # The margins are now calculated dynamically based on the aspect ratio
    aspect_ratio = optimizer.area_width / optimizer.area_length
    # More space on the right for the legend that's now placed outside
    fig.subplots_adjust(left=0.12, right=0.78, bottom=0.1, top=0.88)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax