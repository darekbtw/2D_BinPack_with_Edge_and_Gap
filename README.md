# Filter Placement Optimizer

A Python application that optimizes the placement of rectangular filters on a two-dimensional surface while adhering to specific gap constraints.

## Overview

This application helps to efficiently place filters in a given area while ensuring:
- Proper spacing between filters
- Required gaps from edges
- Maximum utilization of available space

The tool uses optimization algorithms to find efficient arrangements of filters, maximizing coverage while minimizing the number of filter types used.

## Features

- **Interactive GUI**: Easily input area dimensions, gap requirements, and filter sizes
- **Multiple Optimization Algorithms**:
  - Greedy algorithm with randomized trials
  - Simulated annealing for fine-tuning solutions
- **Adaptive Performance**: Automatically adjusts strategies for large areas
- **Filter Type Selection**: Automatically selects the most efficient subset of filter types
- **Visualization**: Displays the optimized filter placement with detailed statistics

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/filter-placement-optimizer.git
   cd filter-placement-optimizer
   ```

2. Install required dependencies:
   ```
   pip install numpy matplotlib tkinter
   ```

## Usage

Run the application:
```
python main.py
```

### Input Parameters

- **Area Length/Width**: The dimensions of the area where filters will be placed
- **Edge Gap**: Required distance between filters and the edge of the area
- **Filter Gap**: Required distance between adjacent filters
- **Filter Sizes**: Available filter dimensions (length × width)
- **Max Filter Types**: Maximum number of different filter sizes to use (2 or 3)
- **Number of Trials**: More trials may yield better results but take longer

### Workflow

1. Enter the area dimensions and gap requirements
2. Add or remove filter size options
3. Set the number of trials and maximum filter types
4. Click "Run Optimization" to generate the optimal placement
5. View the visualization and statistics in the right panel

## Technical Details

### Files

- `main.py`: Entry point for the application
- `filter_gui.py`: Tkinter-based GUI implementation
- `filter_optimizer.py`: Core optimization algorithms and placement logic
- `filter_visualization.py`: Visualization of the filter placement solution

### Optimization Process

1. **Initialize Parameters**: Set up the optimization constraints
2. **Filter Type Selection**: If there are more filter types than the maximum allowed, select the most efficient subset
3. **Greedy Optimization**: Place filters one by one in positions that maximize coverage
4. **Multiple Trials**: Run multiple trials with randomized filter order to find the best solution
5. **Simulated Annealing (Optional)**: Fine-tune the solution with probabilistic adjustments

### Performance Optimizations

- **Adaptive Resolution**: Automatically adjusts grid resolution for large areas
- **Vectorized Operations**: Uses NumPy's vectorized operations for efficient computations
- **Strategic Sampling**: Uses intelligent sampling for extremely large areas
- **Scaling**: Normalizes dimensions to manageable ranges for improved performance

## Example Output

The visualization shows:
- Placed filters with type identification
- Unused areas highlighted in red
- Coverage statistics
- Filter type counts and dimensions
- Required gaps marked with dotted lines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Feel free to contact me if you do decide to add on.