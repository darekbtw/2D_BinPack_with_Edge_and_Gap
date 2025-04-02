import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from filter_optimizer import FilterPlacementOptimizer
from filter_visualization import visualize_solution

class FilterPlacementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Filter Placement Optimizer")
        self.root.geometry("1200x800")
        
        # Default values
        self.area_length = 4000
        self.area_width = 5000
        self.edge_gap = 300
        self.filter_gap = 200
        self.filter_sizes = [
            (457, 457),
            (457, 610),
            (457, 762),
            (457, 915),
            (610, 610),
            (610, 762),
            (610, 915),
            (762, 762),
            (762, 915), 
            (915, 915)
        ]
        self.max_filter_types = 3  # Default max filter types
        
        # Initialize optimizer and solutions
        self.optimizer = None
        self.solutions = []  # List of (solution, coverage, filter_count, filter_variety) tuples
        self.current_solution_index = 0
        
        # Create frames
        self.create_frames()
        self.create_parameter_inputs()
        self.create_filter_size_controls()
        self.create_action_buttons()
        self.create_plot_area()
        self.create_solution_navigation()
        
    def create_frames(self):
        # Left panel for inputs
        self.left_frame = ttk.Frame(self.root, padding=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel for plot
        self.right_frame = ttk.Frame(self.root, padding=10)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Parameters frame
        self.param_frame = ttk.LabelFrame(self.left_frame, text="Area Parameters", padding=10)
        self.param_frame.pack(fill=tk.X, pady=5)
        
        # Filter sizes frame
        self.filter_frame = ttk.LabelFrame(self.left_frame, text="Filter Sizes", padding=10)
        self.filter_frame.pack(fill=tk.X, pady=5)
        
        # Action buttons frame
        self.action_frame = ttk.LabelFrame(self.left_frame, text="Actions", padding=10)
        self.action_frame.pack(fill=tk.X, pady=5)
        
    def create_parameter_inputs(self):
        # Area Length
        ttk.Label(self.param_frame, text="Area Length:").grid(column=0, row=0, sticky=tk.W, pady=2)
        self.area_length_var = tk.DoubleVar(value=self.area_length)
        ttk.Entry(self.param_frame, textvariable=self.area_length_var, width=10).grid(column=1, row=0, pady=2)
        
        # Area Width
        ttk.Label(self.param_frame, text="Area Width:").grid(column=0, row=1, sticky=tk.W, pady=2)
        self.area_width_var = tk.DoubleVar(value=self.area_width)
        ttk.Entry(self.param_frame, textvariable=self.area_width_var, width=10).grid(column=1, row=1, pady=2)
        
        # Edge Gap
        ttk.Label(self.param_frame, text="Edge Gap:").grid(column=0, row=2, sticky=tk.W, pady=2)
        self.edge_gap_var = tk.DoubleVar(value=self.edge_gap)
        ttk.Entry(self.param_frame, textvariable=self.edge_gap_var, width=10).grid(column=1, row=2, pady=2)
        
        # Filter Gap
        ttk.Label(self.param_frame, text="Filter Gap:").grid(column=0, row=3, sticky=tk.W, pady=2)
        self.filter_gap_var = tk.DoubleVar(value=self.filter_gap)
        ttk.Entry(self.param_frame, textvariable=self.filter_gap_var, width=10).grid(column=1, row=3, pady=2)
        
        # Max Filter Types
        ttk.Label(self.param_frame, text="Max Filter Types:").grid(column=0, row=4, sticky=tk.W, pady=2)
        self.max_filter_types_var = tk.IntVar(value=self.max_filter_types)
        filter_types_frame = ttk.Frame(self.param_frame)
        filter_types_frame.grid(column=1, row=4, pady=2)
        
        # Radio buttons for max filter types
        ttk.Radiobutton(filter_types_frame, text="Less variations", variable=self.max_filter_types_var, value=2).pack(side=tk.LEFT)
        ttk.Radiobutton(filter_types_frame, text="More Optimial", variable=self.max_filter_types_var, value=3).pack(side=tk.LEFT)
        
    def create_filter_size_controls(self):
        # Add explanation text
        explanation_text = "Add filters below"
        explanation_label = ttk.Label(
            self.filter_frame, 
            text=explanation_text,
            foreground="blue", 
            font=("", 9, "bold")
        )
        explanation_label.grid(column=0, row=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        
        # Headers
        ttk.Label(self.filter_frame, text="Length").grid(column=1, row=1)
        ttk.Label(self.filter_frame, text="Width").grid(column=2, row=1)
        
        # Frame for filter list with scrollbar
        filter_list_frame = ttk.Frame(self.filter_frame)
        filter_list_frame.grid(column=0, row=2, columnspan=4, sticky=tk.NSEW)
        
        # Canvas and scrollbar for filter sizes
        self.filter_canvas = tk.Canvas(filter_list_frame, height=200)
        self.filter_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(filter_list_frame, orient=tk.VERTICAL, command=self.filter_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.filter_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Frame inside canvas for filter entries
        self.filter_entries_frame = ttk.Frame(self.filter_canvas)
        self.filter_canvas_window = self.filter_canvas.create_window((0, 0), window=self.filter_entries_frame, anchor='nw')
        
        self.filter_entries_frame.bind("<Configure>", self.on_filter_frame_configure)
        self.filter_canvas.bind("<Configure>", self.on_filter_canvas_configure)
        
        # Button frame
        buttons_frame = ttk.Frame(self.filter_frame)
        buttons_frame.grid(column=0, row=3, columnspan=4, sticky=tk.W, pady=5)
        
        # Button to add a new filter
        add_btn = ttk.Button(buttons_frame, text="Add Filter Option", command=self.add_filter_entry)
        add_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Populate with existing filter sizes
        self.filter_entries = []
        for length, width in self.filter_sizes:
            self.add_filter_entry(length, width)
            
    def optimize_to_two_filters(self):
        """Optimize the filter types to just 2 types if there are more."""
        if len(self.filter_entries) <= 2:
            messagebox.showinfo("Filter Types", "Already using 2 or fewer filter types.")
            return
            
        # Confirm with user
        if not messagebox.askyesno("Optimize Filters", 
                                 "This will reduce your filter types to the 2 largest ones. Continue?"):
            return
            
        # Get current filter sizes
        current_filters = []
        for entry in self.filter_entries:
            length = entry['length_var'].get()
            width = entry['width_var'].get()
            area = length * width
            current_filters.append((length, width, area))
            
        # Sort by area (largest first)
        current_filters.sort(key=lambda x: x[2], reverse=True)
        
        # Keep only the two largest
        keep_filters = current_filters[:2]
        
        # Clear all entries
        for entry in self.filter_entries:
            entry['frame'].destroy()
        self.filter_entries = []
        
        # Add back the two largest
        for length, width, _ in keep_filters:
            self.add_filter_entry(length, width)
            
        messagebox.showinfo("Filter Types", "Reduced to 2 filter types based on size.")
            
    def on_filter_frame_configure(self, event):
        # Update the scrollregion when the frame size changes
        self.filter_canvas.configure(scrollregion=self.filter_canvas.bbox("all"))
            
    def on_filter_canvas_configure(self, event):
        # Resize the inner frame when the canvas changes size
        self.filter_canvas.itemconfig(self.filter_canvas_window, width=event.width)
        
    def add_filter_entry(self, length=None, width=None):
        row = len(self.filter_entries)
        
        # Create variables
        length_var = tk.DoubleVar(value=length if length is not None else 10)
        width_var = tk.DoubleVar(value=width if width is not None else 8)
        
        # Create widgets
        entry_frame = ttk.Frame(self.filter_entries_frame)
        entry_frame.pack(fill=tk.X, pady=2)
        
        filter_label = ttk.Label(entry_frame, text=f"Option {row+1}:")
        filter_label.pack(side=tk.LEFT, padx=5)
        
        length_entry = ttk.Entry(entry_frame, textvariable=length_var, width=8)
        length_entry.pack(side=tk.LEFT, padx=5)
        
        width_entry = ttk.Entry(entry_frame, textvariable=width_var, width=8)
        width_entry.pack(side=tk.LEFT, padx=5)
        
        delete_btn = ttk.Button(entry_frame, text="X", width=2, 
                                command=lambda f=entry_frame: self.delete_filter_entry(f))
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Store references
        self.filter_entries.append({
            'frame': entry_frame,
            'length_var': length_var,
            'width_var': width_var,
            'delete_btn': delete_btn,
            'label': filter_label
        })
        
    def delete_filter_entry(self, frame):
        # Find and remove the entry
        for i, entry in enumerate(self.filter_entries):
            if entry['frame'] == frame:
                entry['frame'].destroy()
                self.filter_entries.pop(i)
                break
                
        # Update filter numbers
        for i, entry in enumerate(self.filter_entries):
            entry['label'].config(text=f"Option {i+1}:")
            
    def create_action_buttons(self):
        # Trial count frame
        trials_frame = ttk.Frame(self.action_frame)
        trials_frame.pack(fill=tk.X, pady=5)
        
        # Trial count label and entry
        ttk.Label(trials_frame, text="Number of Trials:").pack(side=tk.LEFT, padx=(0, 5))
        self.trials_var = tk.IntVar(value=50)  # Default is 50 trials
        trials_entry = ttk.Entry(trials_frame, textvariable=self.trials_var, width=5)
        trials_entry.pack(side=tk.LEFT, padx=5)
        
        # Trial count explanation
        ttk.Label(trials_frame, text="(more trials = better results but slower)", 
                font=("", 8), foreground="gray").pack(side=tk.LEFT, padx=5)
        
        # Run optimization button
        ttk.Button(self.action_frame, text="Run Optimization", 
                command=self.run_optimization).pack(fill=tk.X, pady=5)
        
        # Filter type explanation
        filter_constraint_frame = ttk.Frame(self.action_frame)
        filter_constraint_frame.pack(fill=tk.X, pady=10)
        
        constraint_text = "Please wait for the trials to complete"
        constraint_label = ttk.Label(filter_constraint_frame, text=constraint_text, 
                                foreground="blue", font=("", 9))
        constraint_label.pack(anchor=tk.W)
        
    def create_plot_area(self):
        # Create matplotlib figure and canvas
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial plot
        self.ax.set_title("Filter Placement Visualization\n(Run Optimization to See Results)")
        self.ax.set_xlabel("Length")
        self.ax.set_ylabel("Width")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Set equal aspect ratio so that the visualization is to scale
        self.ax.set_aspect('equal')
        
        # Adjust margins to provide more space for labels
        self.fig.subplots_adjust(left=0.12, right=0.95, bottom=0.12, top=0.9)
        
        # Draw the canvas
        self.canvas.draw()
        
    def create_solution_navigation(self):
        """Create navigation controls for multiple solutions."""
        # Navigation frame
        self.nav_frame = ttk.Frame(self.right_frame)
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        # Previous button
        self.prev_btn = ttk.Button(self.nav_frame, text="← Previous", command=self.show_previous_solution)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        # Solution counter
        self.solution_counter = ttk.Label(self.nav_frame, text="Solution 0/0")
        self.solution_counter.pack(side=tk.LEFT, padx=10)
        
        # Next button
        self.next_btn = ttk.Button(self.nav_frame, text="Next →", command=self.show_next_solution)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # Solution info label
        self.solution_info = ttk.Label(self.nav_frame, text="")
        self.solution_info.pack(side=tk.RIGHT, padx=10)
        
        # Initially disable navigation
        self.update_navigation_controls()
        
    def update_navigation_controls(self):
        """Update the state of navigation controls based on available solutions."""
        num_solutions = len(self.solutions)
        
        # Update solution counter
        if num_solutions > 0:
            self.solution_counter.config(text=f"Solution {self.current_solution_index + 1}/{num_solutions}")
        else:
            self.solution_counter.config(text="No solutions")
            
        # Update navigation buttons
        self.prev_btn.config(state=tk.NORMAL if self.current_solution_index > 0 else tk.DISABLED)
        self.next_btn.config(state=tk.NORMAL if self.current_solution_index < num_solutions - 1 else tk.DISABLED)
        
        # Update solution info
        if num_solutions > 0:
            solution, coverage, filter_count, filter_variety = self.solutions[self.current_solution_index]
            info_text = f"Coverage: {coverage:.1%} | Filters: {filter_count} | Types: {filter_variety}"
            self.solution_info.config(text=info_text)
        else:
            self.solution_info.config(text="")
            
    def show_previous_solution(self):
        """Show the previous solution if available."""
        if self.current_solution_index > 0:
            self.current_solution_index -= 1
            self.update_visualization()
            self.update_navigation_controls()
            
    def show_next_solution(self):
        """Show the next solution if available."""
        if self.current_solution_index < len(self.solutions) - 1:
            self.current_solution_index += 1
            self.update_visualization()
            self.update_navigation_controls()
            
    def update_visualization(self):
        """Update the visualization with the current solution."""
        if self.solutions and 0 <= self.current_solution_index < len(self.solutions):
            solution, _, _, _ = self.solutions[self.current_solution_index]
            visualize_solution(
                optimizer=self.optimizer,
                solution=solution,
                ax=self.ax,
                fig=self.fig
            )
            self.canvas.draw()
            
    def update_parameters_from_ui(self):
        try:
            # Get values from UI inputs
            self.area_length = self.area_length_var.get()
            self.area_width = self.area_width_var.get()
            self.edge_gap = self.edge_gap_var.get()
            self.filter_gap = self.filter_gap_var.get()
            self.max_filter_types = self.max_filter_types_var.get()
            
            # Get filter sizes
            self.filter_sizes = []
            for entry in self.filter_entries:
                length = entry['length_var'].get()
                width = entry['width_var'].get()
                if length > 0 and width > 0:
                    self.filter_sizes.append((length, width))
            
            return True
            
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid parameter values: {str(e)}")
            return False
            
    def run_optimization(self):
        if not self.update_parameters_from_ui():
            return
            
        if not self.filter_sizes:
            messagebox.showwarning("Warning", "No filter sizes defined. Please add at least one filter size.")
            return
        
        # Get the number of trials from the UI
        try:
            trials = self.trials_var.get()
            if trials <= 0:
                messagebox.showwarning("Warning", "Number of trials must be positive. Using default of 50.")
                trials = 50
        except:
            messagebox.showwarning("Warning", "Invalid number of trials. Using default of 50.")
            trials = 50
        
        # Apply scaling to normalize dimensions to hundreds range
        scaled = self.apply_scaling()
        
        # Create a new optimizer with updated parameters
        self.optimizer = FilterPlacementOptimizer(
            area_length=self.area_length,
            area_width=self.area_width,
            edge_gap=self.edge_gap,
            filter_gap=self.filter_gap,
            filter_sizes=self.filter_sizes,
            max_filter_types=self.max_filter_types_var.get()
        )
        
        # Calculate area size for progress messaging
        area_size = self.area_length * self.area_width
        
        # Create and configure progress window
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Optimization in Progress")
        progress_window.geometry("400x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Set up progress display
        if area_size > 100000:
            progress_text = (
                f"Running 5 complete optimizations with {trials} trials each...\n"
                f"Using adaptive algorithms for better performance.\n"
                f"Please wait, this may take longer than usual."
            )
        else:
            progress_text = f"Running 5 complete optimizations with {trials} trials each...\nPlease wait..."
        
        progress_label = ttk.Label(progress_window, text=progress_text)
        progress_label.pack(pady=10)
        
        # Add a more detailed status label
        status_label = ttk.Label(progress_window, text="Starting optimization...")
        status_label.pack(pady=5)
        
        # Add a counter for current trial
        trial_label = ttk.Label(progress_window, text="Run: 0/5, Trial: 0/0")
        trial_label.pack(pady=5)
        
        # Add a progress bar
        progress = ttk.Progressbar(progress_window, mode='determinate', maximum=trials * 5)  # 5 complete runs
        progress.pack(fill=tk.X, padx=20, pady=10)
        
        # Cancel button
        cancel_var = tk.BooleanVar(value=False)
        
        def cancel_optimization():
            cancel_var.set(True)
            status_label.config(text="Cancelling optimization...")
        
        cancel_button = ttk.Button(progress_window, text="Cancel", command=cancel_optimization)
        cancel_button.pack(pady=10)
        
        # Define progress callback 
        def update_progress(current_trial, total_trials, improved=False, message=None):
            # Update the progress display
            if cancel_var.get():
                return False  # Signal to stop optimization
                
            progress['value'] = current_trial
            
            if message:
                status_label.config(text=message)
            elif improved:
                status_label.config(text=f"Found improved solution in trial {current_trial}")
            
            # Update trial label with run and trial information
            run_number = (current_trial // trials) + 1
            trial_in_run = (current_trial % trials) + 1
            trial_label.config(text=f"Run: {run_number}/5, Trial: {trial_in_run}/{trials}")
            
            # Critical: Update the window to show changes
            progress_window.update_idletasks()
            progress_window.update()
            
            return True  # Continue optimization
        
        # Set the progress callback
        self.optimizer.progress_callback = update_progress
        
        # Use a background thread for optimization to keep UI responsive
        import threading
        
        def run_optimization_thread():
            try:
                self.solutions = self.optimizer.optimize_greedy(num_trials=trials)
                
                # Scale solutions back to original dimensions if scaling was applied
                if scaled and self.solutions:
                    self.solutions = [(self.scale_solution_back(solution), coverage, count, variety)
                                    for solution, coverage, count, variety in self.solutions]
                    
                    # Also restore original dimensions in the optimizer for visualization
                    self.optimizer.area_length = self.original_area_length
                    self.optimizer.area_width = self.original_area_width
                    self.optimizer.edge_gap = self.original_edge_gap
                    self.optimizer.filter_gap = self.original_filter_gap
                    self.optimizer.effective_length = self.original_area_length - 2 * self.original_edge_gap
                    self.optimizer.effective_width = self.original_area_width - 2 * self.original_edge_gap
                    
                    # Scale the grid resolution
                    if hasattr(self.optimizer, 'grid_resolution'):
                        self.optimizer.grid_resolution *= self.scale_factor
                
                # Ensure we're back on the main thread for UI updates
                self.root.after(100, complete_optimization)
            except Exception as e:
                # Handle any exceptions
                error_message = str(e)
                self.root.after(100, lambda: show_error(error_message))
        
        def complete_optimization():
            # Close progress dialog
            if progress_window.winfo_exists():  # Check if window still exists
                progress_window.destroy()
            
            # Reset solution index and update visualization
            self.current_solution_index = 0
            self.update_visualization()
            self.update_navigation_controls()
            
            # Show summary of results
            if self.solutions:
                summary = "Optimization complete!\n\n"
                summary += f"Found {len(self.solutions)} distinct solutions:\n"
                for i, (_, coverage, count, variety) in enumerate(self.solutions, 1):
                    summary += f"\nSolution {i}:\n"
                    summary += f"Coverage: {coverage:.1%}\n"
                    summary += f"Filter Count: {count}\n"
                    summary += f"Filter Types: {variety}\n"
                messagebox.showinfo("Optimization Results", summary)
            else:
                messagebox.showinfo("Result", "No valid solutions found. Try different parameters.")
        
        def show_error(error_message):
            # Close progress dialog
            if progress_window.winfo_exists():  # Check if window still exists
                progress_window.destroy()
            
            messagebox.showerror("Optimization Error", f"An error occurred during optimization:\n{error_message}")
        
        # Start the optimization thread
        optimization_thread = threading.Thread(target=run_optimization_thread)
        optimization_thread.daemon = True  # Thread will close when main program exits
        optimization_thread.start()
            
    def count_filter_types_in_solution(self, solution):
        # Count unique filter types in the solution
        filter_types = set()
        for _, length, width, _, _ in solution:
            # Ensure consistent orientation for counting (longer side first)
            if length < width:
                filter_types.add((width, length))
            else:
                filter_types.add((length, width))
        return len(filter_types)
    
    def apply_scaling(self):
        """
        Apply scaling to inputs to normalize them to the hundreds range.
        This allows the optimizer to work with manageable numbers while
        preserving the user's original dimensions.
        """
        # Store original values for display and final conversion
        self.original_area_length = self.area_length
        self.original_area_width = self.area_width
        self.original_edge_gap = self.edge_gap
        self.original_filter_gap = self.filter_gap
        self.original_filter_sizes = self.filter_sizes.copy()
        
        # Calculate scaling factor - target is to get max dimension to about 100
        max_dimension = max(self.area_length, self.area_width)
        self.scale_factor = 1.0  # Default - no scaling
        
        if max_dimension > 500:
            # Scale down to bring max dimension to around 100-200
            self.scale_factor = max_dimension / 100.0
            
            # Scale all dimensions
            self.area_length = self.area_length / self.scale_factor
            self.area_width = self.area_width / self.scale_factor
            self.edge_gap = self.edge_gap / self.scale_factor
            self.filter_gap = self.filter_gap / self.scale_factor
            
            # Scale filter sizes
            self.filter_sizes = [(length / self.scale_factor, width / self.scale_factor) 
                                for length, width in self.filter_sizes]
            
            print(f"Applied scaling factor of {self.scale_factor:.2f}")
            print(f"Scaled dimensions: {self.area_length:.2f} x {self.area_width:.2f}")
            return True
        
        # No scaling needed
        print("No scaling applied - dimensions are already in a manageable range")
        return False

    def scale_solution_back(self, solution):
        """
        Scale the solution back to the original dimensions.
        """
        if not hasattr(self, 'scale_factor') or self.scale_factor == 1.0:
            return solution  # No scaling to undo
        
        # Scale the solution back to original dimensions
        scaled_solution = []
        for filter_id, length, width, start_x, start_y in solution:
            scaled_solution.append((
                filter_id,
                length * self.scale_factor,  # Scale length back up
                width * self.scale_factor,   # Scale width back up
                start_x * self.scale_factor, # Scale x position back up
                start_y * self.scale_factor  # Scale y position back up
            ))
        
        return scaled_solution