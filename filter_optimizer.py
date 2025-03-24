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
        
        # Effective area after accounting for edge gaps
        self.effective_length = area_length - 2 * edge_gap
        self.effective_width = area_width - 2 * edge_gap
        
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
        end_x = start_x + filter_length
        end_y = start_y + filter_width
        
        # Check if filter fits within the effective area
        if end_x > self.effective_length or end_y > self.effective_width:
            return False
        
        # For large areas, optimize the overlap check with vectorization when possible
        area_size = self.effective_length * self.effective_width
        if area_size > 100000 and hasattr(self, 'grid_resolution') and self.grid_resolution > 1:
            # Convert coordinates to grid space
            grid_start_x = max(0, int(start_x / self.grid_resolution))
            grid_start_y = max(0, int(start_y / self.grid_resolution))
            grid_end_x = min(self.grid_length, int(end_x / self.grid_resolution) + 1)
            grid_end_y = min(self.grid_width, int(end_y / self.grid_resolution) + 1)
            
            # Check overlaps with vectorized operation (much faster for large areas)
            if np.any(grid[grid_start_y:grid_end_y, grid_start_x:grid_end_x] == 1):
                return False
            
            # Calculate gap region to check
            gap_cells = max(1, int(self.filter_gap / self.grid_resolution))
            gap_start_x = max(0, grid_start_x - gap_cells)
            gap_start_y = max(0, grid_start_y - gap_cells)
            gap_end_x = min(self.grid_length, grid_end_x + gap_cells)
            gap_end_y = min(self.grid_width, grid_end_y + gap_cells)
            
            # Create a mask for the filter's area (to exclude it from the gap check)
            filter_mask = np.zeros((self.grid_width, self.grid_length), dtype=bool)
            filter_mask[grid_start_y:grid_end_y, grid_start_x:grid_end_x] = True
            
            # Check gap constraints with vectorization
            gap_region = grid[gap_start_y:gap_end_y, gap_start_x:gap_end_x]
            gap_mask = np.zeros_like(gap_region, dtype=bool)
            gap_mask_y_offset = grid_start_y - gap_start_y
            gap_mask_x_offset = grid_start_x - gap_start_x
            
            # Only create the mask if the offsets are valid
            if (gap_mask_y_offset >= 0 and gap_mask_x_offset >= 0 and
                gap_mask_y_offset + (grid_end_y - grid_start_y) <= gap_region.shape[0] and
                gap_mask_x_offset + (grid_end_x - grid_start_x) <= gap_region.shape[1]):
                gap_mask[gap_mask_y_offset:gap_mask_y_offset+(grid_end_y-grid_start_y), 
                        gap_mask_x_offset:gap_mask_x_offset+(grid_end_x-grid_start_x)] = True
                
                # Check if there are filters in the gap region (excluding the filter area)
                if np.any(np.logical_and(gap_region == 1, ~gap_mask)):
                    return False
            else:
                # Fallback to the non-vectorized check for edge cases
                for x in range(max(0, int(start_x - self.filter_gap)), min(int(self.effective_length), int(end_x + self.filter_gap))):
                    for y in range(max(0, int(start_y - self.filter_gap)), min(int(self.effective_width), int(end_y + self.filter_gap))):
                        # Only check cells outside the current filter
                        if (x < start_x or x >= end_x or y < start_y or y >= end_y):
                            # Only consider it a constraint if there's another filter here
                            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x] == 1:
                                return False
        else:
            # The original non-vectorized approach for smaller areas
            # Check if the space is empty (no overlap with other filters)
            for x in range(int(start_x), int(end_x)):
                for y in range(int(start_y), int(end_y)):
                    if y < grid.shape[0] and x < grid.shape[1] and grid[y, x] == 1:
                        return False
                    
            # Check filter gap constraints
            for x in range(max(0, int(start_x - self.filter_gap)), min(int(self.effective_length), int(end_x + self.filter_gap))):
                for y in range(max(0, int(start_y - self.filter_gap)), min(int(self.effective_width), int(end_y + self.filter_gap))):
                    # Only check cells outside the current filter
                    if (x < start_x or x >= end_x or y < start_y or y >= end_y):
                        # Only consider it a constraint if there's another filter here
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
        
    def optimize_greedy(self, num_trials=100):
        """
        Find a good filter arrangement using a greedy approach with multiple trials.
        
        Parameters:
        -----------
        num_trials : int
            Number of randomized trials to perform
        """
        print(f"Starting greedy optimization with {num_trials} trials...")
        
        # Check for large areas and adjust strategy if needed
        area_size = self.effective_length * self.effective_width
        large_area = area_size > 100000
        
        if large_area:
            print(f"Large area detected: {self.effective_length} x {self.effective_width}")
            print("Using adaptive placement strategy for improved performance")
        
        # If we have more filter types than the maximum allowed, we need to select a subset
        all_filter_sizes = self.filter_sizes.copy()
        best_selected_types = []
        
        if len(all_filter_sizes) > self.max_filter_types:
            print(f"Selecting best {self.max_filter_types} filter types from {len(all_filter_sizes)} options")
            self.selected_filter_types = []  # Reset selected types
            
            # Try different combinations of filter types
            best_subset_coverage = 0
            
            # First try with the target types (target is usually 2)
            target_types = min(2, self.max_filter_types)  # Default target is 2 unless max is less
            
            # Calculate areas for sorting
            areas = [(length*width, (length, width)) for length, width in all_filter_sizes]
            areas.sort(reverse=True)  # Sort by area, largest first
            
            # Try various combinations:
            # 1. The largest 2-3 filter types by area
            # 2. Random combinations
            # 3. A mix of sizes (largest, medium, smallest)
            combinations_to_try = []
            
            # Add the largest types
            for i in range(1, self.max_filter_types + 1):
                subset = [size for _, size in areas[:i]]
                combinations_to_try.append(subset)
            
            # Add some random combinations
            num_random = min(5, 2**len(all_filter_sizes))  # Limit the number of random combinations
            for _ in range(num_random):
                random_subset = random.sample(all_filter_sizes, 
                                            random.randint(1, min(len(all_filter_sizes), self.max_filter_types)))
                combinations_to_try.append(random_subset)
            
            # Add a mix of sizes if we have enough filter types
            if len(areas) >= 3:
                mix_subset = [areas[0][1], areas[len(areas)//2][1], areas[-1][1]]  # Largest, middle, smallest
                combinations_to_try.append(mix_subset[:self.max_filter_types])
            
            # Try each combination
            for subset in combinations_to_try:
                self.filter_sizes = subset
                
                # Run a few trials with this subset
                subset_trials = max(2, num_trials // len(combinations_to_try))
                for trial in range(subset_trials):
                    # Create a grid for the effective area
                    grid = np.zeros((int(self.effective_width), int(self.effective_length)))
                    solution = []
                    
                    # Randomize the order of filter sizes for this trial
                    filter_sizes = self.filter_sizes.copy()
                    random.shuffle(filter_sizes)
                    
                    # Keep trying to place filters until no more can be placed
                    placed_filter = True
                    while placed_filter:
                        placed_filter = False
                        
                        # Try each filter size
                        for filter_length, filter_width in filter_sizes:
                            # Try both orientations
                            for length, width in [(filter_length, filter_width), (filter_width, filter_length)]:
                                # For large areas, use adaptive step sizes to reduce computation
                                area_size = self.effective_length * self.effective_width
                                if area_size > 100000:
                                    # Step size based on filter size and area size 
                                    # For very large areas, use larger step sizes
                                    x_step = max(1, int(length/2))  # Larger step for larger filters
                                    y_step = max(1, int(width/2))   # Larger step for larger filters
                                    
                                    # For truly massive areas (like 10000x8000), use even larger steps
                                    if area_size > 1000000:
                                        x_step = max(x_step, int(length))
                                        y_step = max(y_step, int(width))
                                else:
                                    # Original step size for smaller areas
                                    x_step = max(1, int(length//4))
                                    y_step = max(1, int(width//4))
                                
                                # Scan the grid for possible placement
                                for start_x in range(0, int(self.effective_length - length) + 1, x_step):
                                    for start_y in range(0, int(self.effective_width - width) + 1, y_step):
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
                            if placed_filter:
                                break
                    
                    # Calculate coverage for this solution
                    coverage, _, _ = self._calculate_metrics(solution)
                    
                    # Update the best subset if this one is better
                    if coverage > best_subset_coverage:
                        best_subset_coverage = coverage
                        best_selected_types = subset
                        
                        # Also update the best overall solution if applicable
                        if self._update_best_solution(solution):
                            print(f"  Found improved solution with subset: {subset}")
            
            # Set the selected filter types
            self.selected_filter_types = best_selected_types
            self.filter_sizes = best_selected_types
            
            print(f"Selected filter types: {self.selected_filter_types}")
        else:
            # If we're already within the limit, use all filter types
            self.selected_filter_types = all_filter_sizes
        
        # Now run the main optimization with the selected filter types
        for trial in range(num_trials):
            # Call progress callback if exists
            if self.progress_callback:
                should_continue = self.progress_callback(trial + 1, num_trials)
                if not should_continue:
                    print("Optimization cancelled by user.")
                    break
                    
            if trial % 5 == 0:
                print(f"  Trial {trial+1}/{num_trials}")
                
            # Create a grid for the effective area
            grid = np.zeros((int(self.effective_width), int(self.effective_length)))
            solution = []
            
            # Randomize the order of filter sizes for this trial
            filter_sizes = self.filter_sizes.copy()
            random.shuffle(filter_sizes)
            
            # Keep trying to place filters until no more can be placed
            placed_filter = True
            while placed_filter:
                placed_filter = False
                
                # Try each filter size
                for filter_length, filter_width in filter_sizes:
                    # Try both orientations
                    for length, width in [(filter_length, filter_width), (filter_width, filter_length)]:
                        # Scan the grid for possible placement
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
                    if placed_filter:
                        break
            
            # Update the best solution if this one is better
            was_improved = self._update_best_solution(solution)
            if was_improved:
                print(f"  Found improved solution in trial {trial+1}")
                
                # Call progress callback with improvement flag if exists
                if self.progress_callback:
                    self.progress_callback(trial + 1, num_trials, True)
        
        # Restore the original filter sizes after optimization
        self.filter_sizes = all_filter_sizes
        
        return self.best_solution
    
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
        """Create a mask for tracking required gaps in the effective area.
        
        Parameters:
        -----------
        resolution : float, optional
            Resolution of the grid (smaller values are more precise but use more memory)
            If None, automatically calculates an appropriate resolution based on area size
        """
        # Automatically determine an appropriate resolution based on area size
        if resolution is None:
            # For extremely large areas, use a higher resolution factor
            # This scales the resolution based on the area size
            area_size = self.effective_length * self.effective_width
            if area_size > 1000000:  # Very large area (e.g., 10000 x 8000)
                resolution = max(self.filter_gap, min(self.effective_length, self.effective_width) / 500)
            elif area_size > 100000:  # Large area
                resolution = max(self.filter_gap / 2, min(self.effective_length, self.effective_width) / 300)
            else:  # Small to medium area
                resolution = 1  # Use the original resolution for small areas
                
            # Make sure resolution is at least 1 to avoid excessive memory usage
            resolution = max(1, resolution)
            print(f"Automatically selected resolution: {resolution}")
            
        # Initialize mask grid at the given resolution
        self.grid_resolution = resolution
        self.grid_length = max(1, int(self.effective_length / resolution))
        self.grid_width = max(1, int(self.effective_width / resolution))
        
        # 0 = available space, 1 = filter, 2 = required gap
        self.gap_mask = np.zeros((self.grid_width, self.grid_length), dtype=int)
    
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
    
    # Add these functions to filter_optimizer.py

def _place_filter_strategic(self, grid, length, width, solution):
    """
    Place a filter using a strategic sampling approach for large areas.
    This method samples the grid at strategic locations rather than
    exhaustively checking every position.
    
    Returns True if a filter was placed, False otherwise.
    """
    # For very large areas, we'll use a more strategic approach
    area_size = self.effective_length * self.effective_width
    
    # Calculate appropriate step sizes based on area size and filter dimensions
    if area_size > 10000000:  # Extremely large (e.g., 50000x50000)
        base_step = min(length, width) * 2
    elif area_size > 1000000:  # Very large (e.g., 10000x8000)
        base_step = min(length, width)
    else:  # Large but not extreme
        base_step = min(length, width) / 2
        
    # Ensure minimum step size of 1
    step_size = max(1, int(base_step))
    
    # First try placing at the borders (more efficient packing usually happens at edges)
    border_positions = self._get_border_positions(length, width, step_size)
    
    for start_x, start_y in border_positions:
        if self._can_place_filter(grid, length, width, start_x, start_y):
            grid = self._place_filter(grid, length, width, start_x, start_y)
            solution.append((len(solution) + 1, length, width, start_x, start_y))
            return True
            
    # Next, try a grid-based sampling approach
    for start_x in range(0, int(self.effective_length - length) + 1, step_size):
        # For very large areas, sample rows less frequently too
        y_step = step_size if area_size > 1000000 else max(1, int(step_size/2))
        
        for start_y in range(0, int(self.effective_width - width) + 1, y_step):
            if self._can_place_filter(grid, length, width, start_x, start_y):
                grid = self._place_filter(grid, length, width, start_x, start_y)
                solution.append((len(solution) + 1, length, width, start_x, start_y))
                return True
                
    # If we're still looking, try a more randomized approach for diversity
    if area_size > 1000000:
        return self._place_filter_random(grid, length, width, solution, 30)
                
    # Unable to place a filter
    return False
    
def _get_border_positions(self, length, width, step_size):
    """
    Generate positions along the borders of the effective area.
    These are often good places to start placing filters.
    """
    positions = []
    
    # Left border
    for y in range(0, int(self.effective_width - width) + 1, step_size):
        positions.append((0, y))
        
    # Right border
    right_x = int(self.effective_length - length)
    for y in range(0, int(self.effective_width - width) + 1, step_size):
        positions.append((right_x, y))
        
    # Top border (excluding corners already added)
    for x in range(step_size, right_x, step_size):
        positions.append((x, 0))
        
    # Bottom border (excluding corners already added)
    bottom_y = int(self.effective_width - width)
    for x in range(step_size, right_x, step_size):
        positions.append((x, bottom_y))
        
    return positions
    
def _place_filter_random(self, grid, length, width, solution, attempts=20):
    """
    Try to place a filter at random positions.
    This introduces diversity in the placement for large areas.
    """
    import random
    
    max_x = int(self.effective_length - length)
    max_y = int(self.effective_width - width)
    
    if max_x <= 0 or max_y <= 0:
        return False
        
    for _ in range(attempts):
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)
        
        if self._can_place_filter(grid, length, width, start_x, start_y):
            grid = self._place_filter(grid, length, width, start_x, start_y)
            solution.append((len(solution) + 1, length, width, start_x, start_y))
            return True
            
    return False