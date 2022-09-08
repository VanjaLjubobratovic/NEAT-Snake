from asyncore import write
import sys
from tokenize import Double
from plotter import plot


def parse_data():
    time = 0
    parsed_file = open("./Outputs/neat_output_500_parallel_clean.txt", 'a')
    with open("./Outputs/neat_output_500_parallel.txt") as f:
        for line in f:
            if "Running generation" in line:
                parsed_file.write(line)
            elif "Generation time:" in line:
                time += float(line.replace("Generation time: ", "").split(" ")[0])
                parsed_file.write(str(round(time, 3)) + "\n")

    parsed_file.close()

def data_to_graph():
    fitness_to_plot = []
    mean_fitness_to_plot = []
    with open("./Outputs/neat_output_500_sequential.txt") as f:
        for line in f:
            if "Best fitness: " in line:
                fitness = line.replace("Best fitness: ", "").split(" ")[0]
                fitness_to_plot.append(round(float(fitness), 1))
            elif "Population's average fitness: " in line:
                avg_fitness = line.replace("Population's average fitness: ", "").split(" ")[0]
                mean_fitness_to_plot.append(round(float(avg_fitness), 1))
    
    plot([(fitness_to_plot, "Best score"), (mean_fitness_to_plot, "Mean gen. fitness")], "Generations", "Score", -50, "neat_sequential_500.png")


if __name__ == "__main__":
    input = sys.argv[1].lower()
    print(input)
    if input == "parse":
        print("Parsing data")
        parse_data()
    elif input == "graph":
        print("Drawing graph")
        data_to_graph()