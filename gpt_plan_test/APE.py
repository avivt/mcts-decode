from ReasoningTasks import *

# a class that inherits from ReasoningTasks and evaluates a prompt
class Evaluator(ReasoningTasks):
    def __init__(self, engine, verbose):
        super().__init__(engine, verbose)

    # the text for a recipe
    def get_recipe(self, recipe):
        return f"The recipe for computing a plan is as follows.\n  {recipe}."
    
    # the text to ask for a recipe
    def ask_for_recipe(self):
        return f"Please explain in simple words a logical recipe for how to generate [PLAN] from [STATEMENT]."
        # return f"The recipe for computing a plan is <INSERT>\n "

    # function that evaluates the prompt, by calling ReasoningTasks
    # and returns the output
    def eval_prompt(self, config_file, prompt, ask_for_prompt=False, t1_or_t4="1_reasoning", max_queries=100):
        self.read_config(config_file)

        domain_name = self.data['domain']
        domain_pddl = f'./instances/{self.data["domain_file"]}'
        instance_folder = f'./instances/{domain_name}/'
        instance = f'./instances/{domain_name}/{self.data["instances_template"]}'
        n_files = min(self.data['n_instances'], len(os.listdir(instance_folder)))

        i_start = self.data['start']
        i_end = min(self.data['end'], max_queries)
        n_files = i_end - i_start + 1  # min(self.data['n_instances'], len(os.listdir(instance_folder)))
        final_output = ""
        gpt3_response = ""
        correct_plans = 0
        for start in range(i_start, i_end + 2 - self.n_examples):
            query = self.data["domain_intro"]
            if not ask_for_prompt:
                query += self.get_recipe(prompt)
            for i in range(start, start + self.n_examples + 1):
                last_plan = True if i == start + self.n_examples else False
                get_plan = not last_plan or ask_for_prompt  # if not last_plan or we ask for prompt, then we want to get the plan. Otherwise, LLM will generate the plan
                cur_instance = instance.format(i)
                # --------------- Add to final output --------------- #
                final_output += f"\n Instance {cur_instance}\n"
                if self.verbose:
                    print(f"Instance {cur_instance}")
                # --------------- Read Instance --------------- #
                problem = self.get_problem(cur_instance, domain_pddl)
                # --------------------------------------------- #
                # ------------ Put plan and instance into text ------------ #
                gt_plan = self.compute_plan(domain_pddl, cur_instance)
                gt_plan_text = get_plan_as_text(self.data)
                query += fill_template(*instance_to_text_blocksworld(problem, get_plan, self.data))
                # --------------------------------------------------------- #

            if ask_for_prompt:
                query += self.ask_for_recipe()
            # Querying LLM
            gpt3_response = send_query(query, self.engine, self.max_gpt_response_length, model=self.model)

            if not ask_for_prompt:
                # Do text_to_plan procedure
                _, gpt3_plan = text_to_plan_blocksworld(gpt3_response, problem.actions, self.gpt3_plan_file, self.data)
                # Apply VAL
                correct = int(validate_plan(domain_pddl, cur_instance, self.gpt3_plan_file))
                correct_plans += correct

            final_output += success_template.format('='*35, t1_or_t4, "SUCCESS" if correct else "FAILURE", '='*35)
            final_output += verbose_template.format(query, gpt3_response, gpt3_plan, gt_plan_text, '='*77) if self.verbose else ""
            if self.verbose: print(final_output)

            # self.save_output("task" + t1_or_t4, final_output)

        if ask_for_prompt:
            return gpt3_response
        
        os.remove(self.plan_file)
        os.remove(self.gpt3_plan_file)

        # --------------- Add to final output --------------- #
        final_output += f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%"
        print(f"[+]: The number of correct plans is {correct_plans}/{n_files}={correct_plans / (n_files) * 100}%")
        # self.save_output("task" + t1_or_t4, final_output)
        return final_output


if __name__ == '__main__':
    random.seed(10)
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='davinci', help='Engine to use')
    parser.add_argument('--verbose', type=str, default="False", help='Verbose')
    args = parser.parse_args()
    task = 't1'
    engine = args.engine
    verbose = eval(args.verbose)
    tasks_obj = Evaluator(engine, verbose)
    tasks_obj.n_examples = 3
    config_file = './configs/t1_goal_directed_reasoning.yaml'
    prompt = tasks_obj.eval_prompt(config_file, prompt='', ask_for_prompt=True, max_queries=10)
    prompt = ''
    print(prompt)

