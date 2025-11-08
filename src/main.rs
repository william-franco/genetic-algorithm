use crossterm::{
    ExecutableCommand,
    event::{self, Event, KeyCode},
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use rand::Rng;
use rand::seq::SliceRandom;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
};
use std::io;
use std::time::Duration;

// Individual representation in the genetic algorithm
#[derive(Clone, Debug)]
struct Individual {
    genes: Vec<f64>,
    fitness: f64,
}

impl Individual {
    // Create a new individual with random genes
    fn new(gene_count: usize, min: f64, max: f64) -> Self {
        let mut rng = rand::thread_rng();
        let genes: Vec<f64> = (0..gene_count).map(|_| rng.gen_range(min..max)).collect();

        Individual {
            genes,
            fitness: 0.0,
        }
    }

    // Calculate fitness based on the objective function
    fn calculate_fitness(&mut self, fitness_fn: &dyn Fn(&[f64]) -> f64) {
        self.fitness = fitness_fn(&self.genes);
    }
}

// Genetic Algorithm structure
struct GeneticAlgorithm {
    population_size: usize,
    gene_count: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    elite_count: usize,
    population: Vec<Individual>,
    min_value: f64,
    max_value: f64,
    generation: usize,
    best_individual: Option<Individual>,
}

impl GeneticAlgorithm {
    // Initialize the genetic algorithm
    fn new(
        population_size: usize,
        gene_count: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        elite_count: usize,
        min_value: f64,
        max_value: f64,
    ) -> Self {
        let population: Vec<Individual> = (0..population_size)
            .map(|_| Individual::new(gene_count, min_value, max_value))
            .collect();

        GeneticAlgorithm {
            population_size,
            gene_count,
            mutation_rate,
            crossover_rate,
            elite_count,
            population,
            min_value,
            max_value,
            generation: 0,
            best_individual: None,
        }
    }

    // Evaluate all individuals in the population
    fn evaluate(&mut self, fitness_fn: &dyn Fn(&[f64]) -> f64) {
        for individual in &mut self.population {
            individual.calculate_fitness(fitness_fn);
        }

        // Sort by fitness (descending)
        self.population
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Update best individual
        self.best_individual = Some(self.population[0].clone());
    }

    // Tournament selection: select best from random sample
    fn tournament_selection(&self, tournament_size: usize) -> &Individual {
        let mut rng = rand::thread_rng();
        let tournament: Vec<&Individual> = self
            .population
            .choose_multiple(&mut rng, tournament_size)
            .collect();

        tournament
            .iter()
            .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
            .unwrap()
    }

    // Single-point crossover between two parents
    fn crossover(&self, parent1: &Individual, parent2: &Individual) -> (Individual, Individual) {
        let mut rng = rand::thread_rng();

        // Fixed: use gen_range instead of gen::<f64>()
        if rng.gen_range(0.0..1.0) > self.crossover_rate {
            return (parent1.clone(), parent2.clone());
        }

        let crossover_point = rng.gen_range(1..self.gene_count);

        let mut child1_genes = parent1.genes[..crossover_point].to_vec();
        child1_genes.extend_from_slice(&parent2.genes[crossover_point..]);

        let mut child2_genes = parent2.genes[..crossover_point].to_vec();
        child2_genes.extend_from_slice(&parent1.genes[crossover_point..]);

        (
            Individual {
                genes: child1_genes,
                fitness: 0.0,
            },
            Individual {
                genes: child2_genes,
                fitness: 0.0,
            },
        )
    }

    // Mutate an individual's genes randomly
    fn mutate(&self, individual: &mut Individual) {
        let mut rng = rand::thread_rng();

        for gene in &mut individual.genes {
            // Fixed: use gen_range instead of gen::<f64>()
            if rng.gen_range(0.0..1.0) < self.mutation_rate {
                *gene = rng.gen_range(self.min_value..self.max_value);
            }
        }
    }

    // Evolve to the next generation
    fn evolve(&mut self, fitness_fn: &dyn Fn(&[f64]) -> f64) {
        let mut new_population = Vec::with_capacity(self.population_size);

        // Elitism: keep best individuals
        for i in 0..self.elite_count {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring
        while new_population.len() < self.population_size {
            let parent1 = self.tournament_selection(3);
            let parent2 = self.tournament_selection(3);

            let (mut child1, mut child2) = self.crossover(parent1, parent2);

            self.mutate(&mut child1);
            self.mutate(&mut child2);

            new_population.push(child1);
            if new_population.len() < self.population_size {
                new_population.push(child2);
            }
        }

        self.population = new_population;
        self.generation += 1;
        self.evaluate(fitness_fn);
    }

    // Get statistics about current generation
    fn get_stats(&self) -> (f64, f64, f64) {
        let best = self.population[0].fitness;
        let worst = self.population[self.population_size - 1].fitness;
        let avg =
            self.population.iter().map(|i| i.fitness).sum::<f64>() / self.population_size as f64;
        (best, avg, worst)
    }
}

// Objective function: maximize f(x, y) = -(x^2 + y^2) + 100
// The maximum is at (0, 0) with value 100
fn fitness_function(genes: &[f64]) -> f64 {
    let x = genes[0];
    let y = genes[1];
    -(x * x + y * y) + 100.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // GA parameters
    let population_size = 100;
    let gene_count = 2; // x and y
    let mutation_rate = 0.1;
    let crossover_rate = 0.8;
    let elite_count = 5;
    let max_generations = 200;
    let min_value = -10.0;
    let max_value = 10.0;

    let mut ga = GeneticAlgorithm::new(
        population_size,
        gene_count,
        mutation_rate,
        crossover_rate,
        elite_count,
        min_value,
        max_value,
    );

    // Initial evaluation
    ga.evaluate(&fitness_function);

    let mut history: Vec<(f64, f64, f64)> = Vec::new();

    // Main loop
    loop {
        // Check for key press
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') || key.code == KeyCode::Esc {
                    break;
                }
            }
        }

        // Draw UI - Fixed: use size() instead of area()
        terminal.draw(|f| {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(3),
                    Constraint::Length(3),
                    Constraint::Length(5),
                    Constraint::Length(5),
                    Constraint::Min(10),
                ])
                .split(f.size());

            // Title
            let title = Paragraph::new("Genetic Algorithm - Function Optimization")
                .style(
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
                .block(Block::default().borders(Borders::ALL));
            f.render_widget(title, chunks[0]);

            // Generation info
            let (best, avg, worst) = ga.get_stats();
            let gen_info = Paragraph::new(format!(
                "Generation: {} | Best: {:.4} | Avg: {:.4} | Worst: {:.4}",
                ga.generation, best, avg, worst
            ))
            .style(Style::default().fg(Color::Green))
            .block(Block::default().borders(Borders::ALL).title("Statistics"));
            f.render_widget(gen_info, chunks[1]);

            // Best solution
            if let Some(ref best_ind) = ga.best_individual {
                let solution = Paragraph::new(vec![
                    Line::from(vec![
                        Span::styled("Best Solution: ", Style::default().fg(Color::Yellow)),
                        Span::raw(format!(
                            "x = {:.6}, y = {:.6}",
                            best_ind.genes[0], best_ind.genes[1]
                        )),
                    ]),
                    Line::from(vec![
                        Span::styled("Fitness: ", Style::default().fg(Color::Yellow)),
                        Span::raw(format!("{:.6}", best_ind.fitness)),
                    ]),
                    Line::from(vec![
                        Span::styled("Target: ", Style::default().fg(Color::Yellow)),
                        Span::raw("f(x,y) = -(x² + y²) + 100, Max at (0,0) = 100"),
                    ]),
                ])
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Best Individual"),
                );
                f.render_widget(solution, chunks[2]);
            }

            // Progress bar
            let progress = (ga.generation as f64 / max_generations as f64 * 100.0) as u16;
            let gauge = Gauge::default()
                .block(Block::default().borders(Borders::ALL).title("Progress"))
                .gauge_style(Style::default().fg(Color::Magenta))
                .percent(progress);
            f.render_widget(gauge, chunks[3]);

            // History chart (simple text-based)
            let mut chart_lines = vec![Line::from(Span::styled(
                "Fitness History (Last 20 gens):",
                Style::default().fg(Color::White),
            ))];

            for (i, (b, _a, w)) in history.iter().rev().take(20).rev().enumerate() {
                let bar_length = ((b - w) / 5.0).max(0.0).min(20.0) as usize;
                let bar = "█".repeat(bar_length);
                chart_lines.push(Line::from(format!(
                    "Gen {:-3}: {} {:.2}",
                    ga.generation.saturating_sub(19 - i),
                    bar,
                    b
                )));
            }

            let history_widget = Paragraph::new(chart_lines)
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .title("Evolution Chart"),
                )
                .style(Style::default().fg(Color::Green));
            f.render_widget(history_widget, chunks[4]);
        })?;

        // Evolve if not finished
        if ga.generation < max_generations {
            ga.evolve(&fitness_function);
            let stats = ga.get_stats();
            history.push(stats);

            // Keep history manageable
            if history.len() > 100 {
                history.remove(0);
            }
        }

        std::thread::sleep(Duration::from_millis(100));
    }

    // Cleanup
    disable_raw_mode()?;
    terminal.backend_mut().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    // Print final results
    if let Some(ref best) = ga.best_individual {
        println!("\n=== Final Results ===");
        println!("Generations: {}", ga.generation);
        println!(
            "Best Solution: x = {:.6}, y = {:.6}",
            best.genes[0], best.genes[1]
        );
        println!("Fitness: {:.6}", best.fitness);
        println!("Expected maximum: f(0, 0) = 100");
    }

    Ok(())
}
