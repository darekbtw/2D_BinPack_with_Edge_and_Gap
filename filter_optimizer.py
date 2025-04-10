import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import copy

class FilterPlacementOptimizer:
    def __init__(self, area_length, area_width, edge_gap, filter_gap, filter_sizes, max_filter_types=3):
        """
        Initialize the optimizer with the given parameters.
        
        Parameters:
        -----------
        area_length : float
            Length of the available area
        area_width : float
            Width of the available area
        edge_gap : float
            Required gap between the edge and filters
        filter_gap : float
            Required gap between adjacent filters
        filter_sizes : list of tuples
            List of (length, width) tuples for each filter size
        max_filter_types : int
            Maximum number of filter types to use in optimization (default: 3)
        """
        self.area_length = area_length
        self.area_width = area_width
        self.edge_gap = edge_gap
        self.filter_gap = filter_gap
        self.filter_sizes = filter_sizes
        self.max_filter_types = max_filter_types
        self.selected_filter_types = []  # Will track which filter types were selected for use
        self.max_solutions = 5  # Maximum number of solutions to keep track of
        
        # Effective area after accounting for edge gaps
        self.effective_length = area_length - 2 * edge_gap
        self.effective_width = area_width - 2 * edge_gap
        
        # For tracking multiple solutions
        self.solutions = []  # List of (solution, coverage, filter_count, filter_variety) tuples
        
        # For tracking the best solution
        self.best_solution = None
        self.best_coverage = 0
        self.best_filter_count = float('inf')
        self.best_filter_variety = float('inf')
        
        # Detect large areas
        self.is_large_area = (self.effective_length * self.effective_width) > 100000
        if self.is_large_area:
            print(f"Large area detected ({self.effective_length} x {self.effective_width}). Using optimized algorithms.")
        
        # Create a mask for tracking required gaps
        self.create_gap_mask()
        
        # Progress tracking callbacks
        self.progress_callback = None
        self.annealing_progress_callback = None
        
    def _can_place_filter(self, grid, filter_length, filter_width, start_x, start_y):
        """Check if a filter can be placed at the given position."""
        # Use exact floating-point calculations for boundaries
        end_x = start_x + filter_length
        end_y = start_y + filter_width
        
        # Check if filter fits within the effective area with floating-point precision
        if end_x > self.effective_length + 0.001 or end_y > self.effective_width + 0.001:  # Small epsilon for floating point comparison
            return False
            
        # For large areas, optimize the overlap check with vectorization when possible
        area_size = self.effective_length * self.effective_width
        if area_size > 100000 and hasattr(self, 'grid_resolution') and self.grid_resolution > 1:
            # Convert coordinates to grid space with ceiling for safety
            grid_start_x = max(0, int(np.floor(start_x / self.grid_resolution)))
            grid_start_y = max(0, int(np.floor(start_y / self.grid_resolution)))
            grid_end_x = min(self.grid_length, int(np.ceil(end_x / self.grid_resolution)))
            grid_end_y = min(self.grid_width, int(np.ceil(end_y / self.grid_resolution)))
            
            # Check overlaps
            if np.any(grid[grid_start_y:grid_end_y, grid_start_x:grid_end_x] == 1):
                return False
                
            # Calculate gap region with exact floating-point math
            gap_cells = max(1, int(np.ceil(self.filter_gap / self.grid_resolution)))
            gap_start_x = max(0, grid_start_x - gap_cells)
            gap_start_y = max(0, grid_start_y - gap_cells)
            gap_end_x = min(self.grid_length, grid_end_x + gap_cells)
            gap_end_y = min(self.grid_width, grid_end_y + gap_cells)
            
            # Check gap constraints
            gap_region = grid[gap_start_y:gap_end_y, gap_start_x:gap_end_x]
            filter_area = np.zeros_like(gap_region, dtype=bool)
            local_start_x = grid_start_x - gap_start_x
            local_start_y = grid_start_y - gap_start_y
            local_end_x = local_start_x + (grid_end_x - grid_start_x)
            local_end_y = local_start_y + (grid_end_y - grid_start_y)
            
            if (local_start_x >= 0 and local_start_y >= 0 and 
                local_end_x <= gap_region.shape[1] and local_end_y <= gap_region.shape[0]):
                filter_area[local_start_y:local_end_y, local_start_x:local_end_x] = True
                if np.any(np.logical_and(gap_region == 1, ~filter_area)):
                    return False
            else:
                # Fallback to exact floating-point checks for edge cases
                for x in range(max(0, int(np.floor(start_x - self.filter_gap))), 
                             min(int(self.effective_length), int(np.ceil(end_x + self.filter_gap)))):
                    for y in range(max(0, int(np.floor(start_y - self.filter_gap))),
                                 min(int(self.effective_width), int(np.ceil(end_y + self.filter_gap)))):
                        if (x < start_x - 0.001 or x > end_x + 0.001 or 
                            y < start_y - 0.001 or y > end_y + 0.001):
                            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == 1:
                                return False
        else:
            # For smaller areas, use exact floating-point checks
            for x in range(int(np.floor(start_x)), int(np.ceil(end_x))):
                for y in range(int(np.floor(start_y)), int(np.ceil(end_y))):
                    if y < grid.shape[0] and x < grid.shape[1] and grid[y, x] == 1:
                        return False
                    
            # Check filter gap constraints with exact floating-point math
            for x in range(max(0, int(np.floor(start_x - self.filter_gap))),
                         min(int(self.effective_length), int(np.ceil(end_x + self.filter_gap)))):
                for y in range(max(0, int(np.floor(start_y - self.filter_gap))),
                             min(int(self.effective_width), int(np.ceil(end_y + self.filter_gap)))):
                    if (x < start_x - 0.001 or x > end_x + 0.001 or 
                        y < start_y - 0.001 or y > end_y + 0.001):
                        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == 1:
                            return False
                    
        return True
        
    def _place_filter(self, grid, filter_length, filter_width, start_x, start_y):
        """Place a filter at the given position and return the updated grid."""
        new_grid = grid.copy()
        
        # Check for large areas and adapt placement strategy
        area_size = self.effective_length * self.effective_width
        
        if area_size > 100000 and hasattr(self, 'grid_resolution') and self.grid_resolution > 1:
            # For large areas, use vectorized operations whenever possible
            # This is much faster than iterating pixel by pixel
            
            # Convert to grid coordinates
            grid_start_x = max(0, int(start_x / self.grid_resolution))
            grid_start_y = max(0, int(start_y / self.grid_resolution))
            grid_end_x = min(self.grid_length, int((start_x + filter_length) / self.grid_resolution) + 1)
            grid_end_y = min(self.grid_width, int((start_y + filter_width) / self.grid_resolution) + 1)
            
            # Set all cells in the filter area to 1 (vectorized operation)
            if (0 <= grid_start_y < new_grid.shape[0] and 
                0 <= grid_start_x < new_grid.shape[1] and
                0 <= grid_end_y <= new_grid.shape[0] and 
                0 <= grid_end_x <= new_grid.shape[1]):
                new_grid[grid_start_y:grid_end_y, grid_start_x:grid_end_x] = 1
        else:
            # Original approach for smaller areas
            for x in range(int(start_x), int(start_x + filter_length)):
                for y in range(int(start_y), int(start_y + filter_width)):
                    if 0 <= y < new_grid.shape[0] and 0 <= x < new_grid.shape[1]:
                        new_grid[y, x] = 1
                        
        return new_grid
    
    def _calculate_metrics(self, solution):
        """Calculate coverage, filter count, and variety for a solution.
        
        This calculates coverage as the ratio of area covered by filters to 
        the total available area (excluding required gaps).
        """
        if not solution:
            return 0, 0, 0
            
        # Update the gap mask for this solution
        self.update_gap_mask(solution)
        
        # Calculate how much area is covered by filters
        filter_cells = np.sum(self.gap_mask == 1)
        
        # Calculate how much area is required gaps
        gap_cells = np.sum(self.gap_mask == 2)
        
        # Calculate how much area is available (total area - required gaps)
        total_cells = self.grid_width * self.grid_length
        available_cells = total_cells - gap_cells
        
        # Calculate coverage as proportion of available area
        if available_cells > 0:
            coverage = filter_cells / available_cells
        else:
            coverage = 0
        
        # Count filters
        filter_count = len(solution)
        
        # Count unique filter types
        filter_types = set()
        for _, length, width, _, _ in solution:
            # Ensure consistent orientation for counting (longer side first)
            if length < width:
                filter_types.add((width, length))
            else:
                filter_types.add((length, width))
        filter_variety = len(filter_types)
        
        return coverage, filter_count, filter_variety
    
    def _is_better_solution(self, new_solution):
        """Determine if the new solution is better than the current best."""
        if not new_solution:
            return False
            
        new_coverage, new_count, new_variety = self._calculate_metrics(new_solution)
        
        # If we have no solution yet, this is better
        if self.best_solution is None:
            return True
            
        # Prioritize coverage first
        if new_coverage > self.best_coverage + 0.05:  # 5% better coverage is significant
            return True
            
        # If coverage is similar, prioritize fewer filters
        if abs(new_coverage - self.best_coverage) < 0.05 and new_count < self.best_filter_count:
            return True
            
        # If coverage and filter count are similar, prioritize less variety
        if (abs(new_coverage - self.best_coverage) < 0.05 and 
            new_count <= self.best_filter_count + 1 and 
            new_variety < self.best_filter_variety):
            return True
            
        return False
        
    def _update_best_solution(self, solution):
        """Update the best solution if the new one is better."""
        if self._is_better_solution(solution):
            self.best_solution = solution
            self.best_coverage, self.best_filter_count, self.best_filter_variety = self._calculate_metrics(solution)
            return True
        return False
        
    def _center_solution(self, solution):
        """
        Center the entire solution by moving all filters as a group.
        This distributes unused space evenly around the edges.
        """
        if not solution:
            return solution

        # Find the current bounds of the solution
        min_x = float('inf')
        max_x = float('-inf')
        min_y = float('inf')
        max_y = float('-inf')

        for _, length, width, start_x, start_y in solution:
            min_x = min(min_x, start_x)
            max_x = max(max_x, start_x + length)
            min_y = min(min_y, start_y)
            max_y = max(max_y, start_y + width)

        # Calculate the current size of the used area
        used_width = max_y - min_y
        used_length = max_x - min_x

        # Calculate the offsets needed to center
        x_offset = (self.effective_length - used_length) / 2 - min_x
        y_offset = (self.effective_width - used_width) / 2 - min_y

        # Create new centered solution
        centered_solution = []
        for filter_id, length, width, start_x, start_y in solution:
            new_x = start_x + x_offset
            new_y = start_y + y_offset
            centered_solution.append((filter_id, length, width, new_x, new_y))

        return centered_solution

    def optimize_greedy(self, num_trials=100):
        """
        Run multiple complete optimizations and find the best distinct solutions across all runs.
        
        Parameters:
        -----------
        num_trials : int
            Number of trials per optimization run
            
        Returns:
        --------
        list of tuples
            List of (solution, coverage, filter_count, filter_variety) tuples
        """
        # Number of complete optimization runs
        num_runs = 5
        all_solutions = []  # Store solutions from all runs
        
        # Run complete optimization multiple times
        for run in range(num_runs):
            if self.progress_callback:
                self.progress_callback(run * num_trials, num_runs * num_trials, 
                                    improved=False, 
                                    message=f"Starting optimization run {run + 1}/{num_runs}")
            
            # Reset solutions for this run
            run_solutions = []
            
            # Run trials for this optimization
            for trial in range(num_trials):
                if self.progress_callback:
                    overall_progress = run * num_trials + trial
                    if not self.progress_callback(overall_progress, num_runs * num_trials,
                                                message=f"Run {run + 1}/{num_runs}, Trial {trial + 1}/{num_trials}"):
                        break
                
                # Create a new solution
                solution = self._create_single_solution()
                
                if solution:
                    # Center the solution
                    solution = self._center_solution(solution)
                    metrics = self._calculate_metrics(solution)
                    run_solutions.append((solution, *metrics))
                    
                    if self.progress_callback:
                        self.progress_callback(overall_progress, num_runs * num_trials, 
                                            improved=True,
                                            message=f"Run {run + 1}/{num_runs}: Found solution with {metrics[0]:.1%} coverage")
            
            # Add the best solutions from this run to the overall solutions
            if run_solutions:
                # Sort solutions from this run
                run_solutions.sort(key=lambda x: (x[1], -x[2], -x[3]), reverse=True)
                # Add the best solution from this run to all solutions
                all_solutions.append(run_solutions[0])
        
        # Sort all collected solutions by multiple criteria
        all_solutions.sort(key=lambda x: (x[1], -x[2], -x[3]), reverse=True)
        
        # Select the best distinct solutions
        self.solutions = []
        seen_coverages = set()
        
        for solution, coverage, count, variety in all_solutions:
            # Round coverage to 3 decimal places to group similar solutions
            rounded_coverage = round(coverage, 3)
            
            # Only add if we haven't seen a very similar coverage
            if rounded_coverage not in seen_coverages:
                self.solutions.append((solution, coverage, count, variety))
                seen_coverages.add(rounded_coverage)
                
                # Stop after finding max_solutions distinct solutions
                if len(self.solutions) >= self.max_solutions:
                    break
        
        return self.solutions
    
    def optimize_simulated_annealing(self, initial_temp=100, cooling_rate=0.8, iterations=50, max_temp_levels=10):
        """
        Find an optimal filter arrangement using simulated annealing.
        
        Parameters:
        -----------
        initial_temp : float
            Initial temperature for simulated annealing
        cooling_rate : float
            Rate at which temperature decreases
        iterations : int
            Number of iterations at each temperature level
        max_temp_levels : int
            Maximum number of temperature levels before stopping
        """
        print("Starting simulated annealing optimization...")
        
        # Start with a greedy solution
        if self.best_solution is None:
            self.optimize_greedy(num_trials=5)
        current_solution = copy.deepcopy(self.best_solution)
        
        if not current_solution:
            print("No initial solution found. Exiting simulated annealing.")
            return None
            
        # Center the initial solution
        current_solution = self._center_solution(current_solution)
        current_coverage, current_count, current_variety = self._calculate_metrics(current_solution)
        print(f"Initial solution: Coverage={current_coverage*100:.2f}%, Filters={current_count}, Types={current_variety}")
        
        # For tracking progress
        no_improvement_count = 0
        max_no_improvement = 3  # Stop early if no improvement for this many temp levels
        best_coverage_so_far = current_coverage
        
        # Simulated annealing
        temp = initial_temp
        temp_level = 0
        
        while temp > 0.1 and no_improvement_count < max_no_improvement and temp_level < max_temp_levels:
            temp_level += 1
            print(f"Temperature level {temp_level}/{max_temp_levels}: T = {temp:.2f}")
            
            improvements = 0
            
            for i in range(iterations):
                if i % 10 == 0:
                    print(f"  Iteration {i+1}/{iterations}", end="\r")
                
                # Create a neighboring solution by modifying the current one
                new_solution = self._generate_neighbor(current_solution)
                
                # Center the new solution
                new_solution = self._center_solution(new_solution)
                
                # Calculate new metrics
                new_coverage, new_count, new_variety = self._calculate_metrics(new_solution)
                
                # Calculate the acceptance probability
                delta = ((new_coverage - current_coverage) * 10 - 
                         (new_count - current_count) * 0.1 - 
                         (new_variety - current_variety) * 0.5)
                
                # Accept with probability based on delta and temperature
                if delta > 0 or random.random() < np.exp(delta / temp):
                    current_solution = new_solution
                    current_coverage = new_coverage
                    current_count = new_count
                    current_variety = new_variety
                    
                    # Update best solution if this one is better
                    if self._update_best_solution(current_solution):
                        improvements += 1
                        best_coverage_so_far = max(best_coverage_so_far, current_coverage)
                        print(f"  Found improved solution: Coverage={current_coverage*100:.2f}%, Filters={current_count}, Types={current_variety}")
            
            print(f"  Completed {iterations} iterations at temperature {temp:.2f}")
            
            # Track if we made any improvements at this temperature
            if improvements > 0:
                no_improvement_count = 0
                print(f"  Made {improvements} improvements at this temperature")
            else:
                no_improvement_count += 1
                print(f"  No improvements at this temperature. ({no_improvement_count}/{max_no_improvement})")
            
            # Cool down
            temp *= cooling_rate
        
        print("Simulated annealing complete.")
        return self.best_solution
        
    def _generate_neighbor(self, solution):
        """Generate a neighboring solution for simulated annealing."""
        if not solution:
            return []
            
        # Create a copy of the solution
        new_solution = copy.deepcopy(solution)
        
        # 50% chance to remove a filter and try to add new ones
        # 50% chance to just try to add new filters to empty spaces
        if random.random() < 0.5 and len(new_solution) > 0:
            # Remove 1 random filter
            del new_solution[random.randint(0, len(new_solution)-1)]
        
        # Rebuild the grid
        grid = np.zeros((int(self.effective_width), int(self.effective_length)))
        for filter_id, length, width, start_x, start_y in new_solution:
            grid = self._place_filter(grid, length, width, start_x, start_y)
        
        # Try to add new filters
        filter_sizes = self.filter_sizes.copy()
        random.shuffle(filter_sizes)
        
        # Try to place just a few filters (limited attempts for speed)
        attempt_count = 0
        max_attempts = 10
        
        while attempt_count < max_attempts:
            attempt_count += 1
            placed_filter = False
            
            # Select a random filter size and orientation
            filter_length, filter_width = random.choice(filter_sizes)
            if random.random() < 0.5:  # 50% chance to rotate
                filter_length, filter_width = filter_width, filter_length
            
            # Try a few random positions
            for _ in range(5):  # Try 5 random positions
                # Pick a random position
                max_x = max(0, int(self.effective_length - filter_length))
                max_y = max(0, int(self.effective_width - filter_width))
                
                if max_x > 0 and max_y > 0:
                    start_x = random.randint(0, max_x)
                    start_y = random.randint(0, max_y)
                    
                    if self._can_place_filter(grid, filter_length, filter_width, start_x, start_y):
                        # Place the filter
                        grid = self._place_filter(grid, filter_length, filter_width, start_x, start_y)
                        new_solution.append((len(new_solution) + 1, filter_length, filter_width, start_x, start_y))
                        placed_filter = True
                        break
            
            if not placed_filter:
                # If we couldn't place this filter, try the next size
                continue
        
        # Renumber filters
        for i in range(len(new_solution)):
            filter_id, length, width, start_x, start_y = new_solution[i]
            new_solution[i] = (i + 1, length, width, start_x, start_y)
        
        return new_solution
    
    def create_gap_mask(self, resolution=None):
        """Create a mask for tracking required gaps in the effective area."""
        # For this specific case with filter gaps and edge gaps, use a resolution
        # that's a factor of both to avoid rounding errors
        if resolution is None:
            # Use unit resolution (1) for precise placement
            # This ensures we maintain exact dimensions without scaling
            resolution = 1
        
        # Initialize mask grid at the given resolution
        self.grid_resolution = resolution
        self.grid_length = max(1, int(np.ceil(self.effective_length)))
        self.grid_width = max(1, int(np.ceil(self.effective_width)))
        
        # 0 = available space, 1 = filter, 2 = required gap
        self.gap_mask = np.zeros((self.grid_width, self.grid_length), dtype=int)
        
        # Print actual dimensions being used
        print(f"\nActual dimensions:")
        print(f"Total area: {self.area_length} x {self.area_width}")
        print(f"Effective area: {self.effective_length} x {self.effective_width}")
        print(f"Edge gap: {self.edge_gap}")
        print(f"Filter gap: {self.filter_gap}")
        print(f"Grid resolution: {self.grid_resolution}")
    
    def update_gap_mask(self, solution):
        """Update the gap mask based on the current filter placement.
        
        This tracks:
        - Placed filters (1)
        - Required gaps around filters (2)
        - Available but unused area (0)
        """
        # Reset the gap mask
        self.gap_mask = np.zeros((self.grid_width, self.grid_length), dtype=int)
        
        # Mark each filter position
        for filter_id, length, width, start_x, start_y in solution:
            # Convert to grid coordinates
            grid_start_x = max(0, int(start_x / self.grid_resolution))
            grid_start_y = max(0, int(start_y / self.grid_resolution))
            grid_end_x = min(self.grid_length, int((start_x + length) / self.grid_resolution))
            grid_end_y = min(self.grid_width, int((start_y + width) / self.grid_resolution))
            
            # Mark the filter itself
            for gx in range(grid_start_x, grid_end_x):
                for gy in range(grid_start_y, grid_end_y):
                    if 0 <= gx < self.grid_length and 0 <= gy < self.grid_width:
                        self.gap_mask[gy, gx] = 1
        
        # Create a copy of the mask to identify filter positions
        filter_positions = np.copy(self.gap_mask)
        
        # Now mark the required gaps in a second pass
        for filter_id, length, width, start_x, start_y in solution:
            # Convert to grid coordinates
            grid_start_x = max(0, int(start_x / self.grid_resolution))
            grid_start_y = max(0, int(start_y / self.grid_resolution))
            grid_end_x = min(self.grid_length, int((start_x + length) / self.grid_resolution))
            grid_end_y = min(self.grid_width, int((start_y + width) / self.grid_resolution))
            
            # Calculate gap size in grid cells
            gap_cells = max(1, int(self.filter_gap / self.grid_resolution))
            
            # Mark all cells around the filter as gaps if they're not already filters
            for gx in range(max(0, grid_start_x - gap_cells), min(self.grid_length, grid_end_x + gap_cells)):
                for gy in range(max(0, grid_start_y - gap_cells), min(self.grid_width, grid_end_y + gap_cells)):
                    # Only mark as gap if:
                    # 1. It's not already a filter
                    # 2. It's outside the current filter
                    # 3. There's at least one other filter in the vicinity (to enforce filter-to-filter gaps)
                    if (self.gap_mask[gy, gx] == 0 and 
                        not (grid_start_x <= gx < grid_end_x and grid_start_y <= gy < grid_end_y)):
                        
                        # Check if this is near another filter (excluding the current one)
                        is_near_another_filter = False
                        
                        # Define the vicinity to check (the gap around this cell)
                        check_start_x = max(0, gx - gap_cells)
                        check_end_x = min(self.grid_length, gx + gap_cells + 1)
                        check_start_y = max(0, gy - gap_cells)
                        check_end_y = min(self.grid_width, gy + gap_cells + 1)
                        
                        # Check if there's another filter in this vicinity
                        for check_x in range(check_start_x, check_end_x):
                            for check_y in range(check_start_y, check_end_y):
                                # Skip cells that are part of the current filter
                                if not (grid_start_x <= check_x < grid_end_x and grid_start_y <= check_y < grid_end_y):
                                    if filter_positions[check_y, check_x] == 1:
                                        is_near_another_filter = True
                                        break
                            if is_near_another_filter:
                                break
                        
                        # Only mark as a gap if it's between two filters
                        if is_near_another_filter:
                            self.gap_mask[gy, gx] = 2
        
        # Debug - print counts to verify
        filter_cells = np.sum(self.gap_mask == 1)
        gap_cells = np.sum(self.gap_mask == 2)
        unused_cells = np.sum(self.gap_mask == 0)
        total_cells = self.grid_width * self.grid_length
        
        return self.gap_mask
    
    def _create_single_solution(self):
        """Create a single solution using the greedy approach."""
        # Create a grid for the effective area (no scaling)
        grid = np.zeros((int(self.effective_width), int(self.effective_length)))
        solution = []
        
        # If we have more filter types than the maximum allowed, we need to select a subset
        all_filter_sizes = self.filter_sizes.copy()
        
        if len(all_filter_sizes) > self.max_filter_types:
            # Randomly select max_filter_types number of filter sizes
            self.selected_filter_types = random.sample(all_filter_sizes, self.max_filter_types)
            filter_sizes = self.selected_filter_types
        else:
            # If we're already within the limit, use all filter types
            self.selected_filter_types = all_filter_sizes
            filter_sizes = all_filter_sizes
        
        # For systematic placement, try placing filters in rows
        # Start from left edge, moving right by filter width + gap
        for filter_length, filter_width in filter_sizes:
            # Try both orientations
            for length, width in [(filter_length, filter_width), (filter_width, filter_length)]:
                # Calculate how many filters could theoretically fit in each row and column
                # Add filter gap to account for spaces between filters
                available_length = self.effective_length + self.filter_gap  # Add one gap because we need N-1 gaps for N filters
                available_width = self.effective_width + self.filter_gap
                
                max_in_row = int(available_length / (length + self.filter_gap))
                max_rows = int(available_width / (width + self.filter_gap))
                
                print(f"\nPlacement calculation for {length}x{width} filter:")
                print(f"Available length: {available_length}")
                print(f"Filter + gap: {length + self.filter_gap}")
                print(f"Calculated max in row: {max_in_row}")
                
                # Try placing filters systematically
                for row in range(max_rows):
                    # Calculate Y position
                    start_y = row * (width + self.filter_gap)
                    
                    # Try placing filters in this row
                    for col in range(max_in_row):
                        # Calculate X position
                        start_x = col * (length + self.filter_gap)
                        
                        # Debug output for first placement attempt in each row
                        if col == 0:
                            print(f"\nAttempting placement in row {row}:")
                            print(f"Start position: ({start_x}, {start_y})")
                            print(f"End position: ({start_x + length}, {start_y + width})")
                        
                        if self._can_place_filter(grid, length, width, start_x, start_y):
                            # Place the filter
                            grid = self._place_filter(grid, length, width, start_x, start_y)
                            solution.append((len(solution) + 1, length, width, start_x, start_y))
                            print(f"Successfully placed filter at ({start_x}, {start_y})")
        
        # If systematic placement didn't work well, try filling remaining spaces
        placed_filter = True
        while placed_filter:
            placed_filter = False
            
            # Try each filter size
            for filter_length, filter_width in filter_sizes:
                # Try both orientations
                for length, width in [(filter_length, filter_width), (filter_width, filter_length)]:
                    # For smaller areas, use exhaustive search for remaining spaces
                    for start_x in range(0, int(self.effective_length - length) + 1):
                        for start_y in range(0, int(self.effective_width - width) + 1):
                            if self._can_place_filter(grid, length, width, start_x, start_y):
                                # Place the filter
                                grid = self._place_filter(grid, length, width, start_x, start_y)
                                solution.append((len(solution) + 1, length, width, start_x, start_y))
                                placed_filter = True
                                break
                        if placed_filter:
                            break
                if placed_filter:
                    break
                    
        return solution
