from problem import bbob, bbob_torch, protein_docking


def construct_problem_set(config, seed):
    problem = config.problem
    if problem in ['bbob', 'bbob-noisy']:
        return bbob.BBOB_Dataset.get_datasets(suit=config.problem,
                                              dim=config.dim,
                                              upperbound=config.upperbound,
                                              train_batch_size=config.train_batch_size,
                                              test_batch_size=config.test_batch_size,
                                              difficulty=config.difficulty,
                                              instance_seed = seed,
                                              mix_dim = config.mix_dim)
    elif problem in ['bbob-torch', 'bbob-noisy-torch']:
        return bbob_torch.BBOB_Dataset_torch.get_datasets(suit=config.problem,
                                                          dim=config.dim,
                                                          upperbound=config.upperbound,
                                                          train_batch_size=config.train_batch_size,
                                                          test_batch_size=config.test_batch_size,
                                                          difficulty=config.difficulty,
                                                          instance_seed = seed,)

    elif problem in ['protein', 'protein-torch']:
        return protein_docking.Protein_Docking_Dataset.get_datasets(version=problem,
                                                                    train_batch_size=config.train_batch_size,
                                                                    test_batch_size=config.test_batch_size,
                                                                    difficulty=config.difficulty,
                                                                    dataset_seed = seed)
    else:
        raise ValueError(problem + ' is not defined!')
