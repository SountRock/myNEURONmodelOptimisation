import numpy as np
import random
import inspect
import csv
from deap import base, creator, tools, algorithms
#from pymoo.indicators.hv import HV
from pymoo.indicators.hv import Hypervolume as HV
import json
import os


'''
Loss функции
'''
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

def mae(y_pred, y_true):
    return np.mean(np.abs(y_pred - y_true))

'''
Функция ошибки, при delta < 1 сильнее штрафует за выбросы
'''
def huber(y_pred, y_true, delta=0.35):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.where(mask, 0.5 * r**2, delta * (abs_r - 0.5 * delta))
    return np.mean(loss)

'''
Нормализация если ЦФ в разных диапазонах
'''
def normalize(arr):
    return np.log1p(arr)

'''
Многокритериальная оптимизация c помощью NSGA-II (Парето-оптимизация) + инициализация начальных параметров
Аргументы:
    target_funcs: список целевых функций
    t_matrix: список массивов времени (оси x ЦФ) для каждой целевой функции
    y_true_matrix: список эталонных данных (ground truth) для каждой целевой функции
    param_bounds: словарь {param_name: (low, high)} для границ параметров. *к сожелению может сделать параметры отрицательными - установить защиту в ЦФ
    loss_func: функция потерь 
    n_gen: число поколений (если None -> определяется автоматически)
    pop_size: размер популяции.
    log_path: путь для логирования в CSV 
    checkpoint_path: путь для сохранения/восстановления состояния
    checkpoint_every: как часто сохранять чекпоинт (в поколениях)
    priorities: веса для каждой ЦФ (штраф на loss)
'''
def run_nsga3_optimization(target_funcs, t_matrix, y_true_matrix, param_bounds: dict,
                          loss_func=mse, n_gen=None, pop_size=120,
                          log_path=None, checkpoint_path=None, checkpoint_every=5,
                          priorities=None):  
    #Проверки входных данных 
    if len(t_matrix) != len(target_funcs) or len(y_true_matrix) != len(target_funcs):
        raise ValueError(f"Все матрицы должны быть согласованных размеров t_matrix={t_matrix.shape} target_funcs={target_funcs.shape}")

    #Сбор уникальных имён всех параметров из функций. Механизмы рефлексии
    all_param_names = []
    for f in target_funcs:
        sig = inspect.signature(f)
        f_params = list(sig.parameters.keys())[1:] #первый аргумент всегда t, пропускаем
        all_param_names.extend(f_params)
    all_param_names = list(dict.fromkeys(all_param_names)) #сохраняем порядок, убираем дубли

    #Проверяем, что для каждого параметра есть границы
    for name in all_param_names:
        if name not in param_bounds:
            raise ValueError(f"Потеряны границы для '{name}'")
    bounds = [param_bounds[name] for name in all_param_names]

    #Если число поколений не задано — берём минимум 80 или 15 * count_params
    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))

    num_objectives = len(target_funcs)

    #Приоритеты целей 
    if priorities is None:
        priorities = [1.0] * num_objectives
    elif len(priorities) != num_objectives:
        raise ValueError(f"Длина приоритетов ({len(priorities)}) должна соотвестовать количеству ЦФ ({num_objectives})")

    #Создаем типы для NSGA-II (многокритериальная минимизация)

    #Определяем типы индивидов и фитнеса для DEAP 
    #FitnessMulti — это класс фитнеса для многокритериальной задачи.
    #weights=(-1.0,) * num_objectives означает, что все цели минимизируются.
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    #Individual — это класс индивида, связывается с фитнесом FitnessMulti.
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    #Инициализируем toolbox
    toolbox = base.Toolbox()
    #Регистрируем генераторы "генов" и задаем границы
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)

    #Определяем способ создания одного индивида:
    #tools.initCycle вызывает указанные генераторы один раз (n=1) и объединяет их результаты в объект класса Individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    
    #Определяем способ создания популяции:
    #tools.initRepeat повторяет генерацию individual нужное число раз и упаковывает их в обычный список
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Многокритериальная функция оценки 
    '''
    #Без нормализации
    def evaluate(ind):
        losses = []
        for t_i, f, y_true in zip(t_matrix, target_funcs, y_true_matrix):
            sig = inspect.signature(f)
            f_param_names = list(sig.parameters.keys())[1:]
            f_args = [ind[all_param_names.index(p)] for p in f_param_names]
            y_pred = f(t_i, *f_args)
            losses.append(loss_func(y_pred, y_true))
        return tuple(losses)
    '''
    #С нормализацией
    def evaluate(ind):
        losses = []
        for i, (t_i, f, y_true) in enumerate(zip(t_matrix, target_funcs, y_true_matrix)):
            sig = inspect.signature(f)
            f_param_names = list(sig.parameters.keys())[1:]
            f_args = [ind[all_param_names.index(p)] for p in f_param_names]
            y_pred = f(t_i, *f_args)

            # Нормализация
            y_pred = normalize(y_pred)
            y_true = normalize(y_true)

            loss_val = loss_func(y_pred, y_true)
            weighted_loss = loss_val * priorities[i]  # Умножаем на приоритет
            losses.append(weighted_loss)
        return tuple(losses)

    toolbox.register("evaluate", evaluate)

    #Операторы генетического алгоритма
    #Комбинирует гены двух родителей и еще раз фиксируем границы
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0) #больше eta - потомки ближе к родителям
    #Мутация генов, фиксация параметров
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2) #вероятность мутации каждого гена
    toolbox.register("select", tools.selNSGA2)

    #Задаём точку отсчёта для hypervolume (чуть выше максимальных значений потерь)
    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    #Восстановление из чекпоинта или инициализация популяции 
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)
        print(f"Checkpoint востановлен на start_gen={start_gen}")
    else:
        population = toolbox.population(n=pop_size)
        #Инициализация fitness для начальной популяции
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind)

    #Логирование
    if log_path and start_gen == 0:
        log_file = open(log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        header = ['generation'] + [f'tf{i+1}_loss' for i in range(num_objectives)] + all_param_names + ['hypervolume']
        csv_writer.writerow(header)
    elif log_path:
        log_file = open(log_path, mode='a', newline='')
        csv_writer = csv.writer(log_file)
    else:
        csv_writer = None

    prev_hv = 0.0

    #Оптимизация
    for gen in range(start_gen, n_gen):
        #Определяем текущий фронт Парето
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits = [ind.fitness.values for ind in front]

        fits_np = np.array(fits)
        current_hv = hv_indicator.do(fits_np)

        #Считаем гиперобъём  фронта
        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        #Если гиперобъём растёт — уменьшаем мутацию, увеличиваем скрещивание (эксплуатация)
        #Если гиперобъём падает или не растёт — наоборот (исследование)
        if hv_change > 0:
            cxpb = min(0.9, 0.6 + 0.3 * hv_change)  #Скрещивание
            mutpb = max(0.1, 0.4 - 0.3 * hv_change) #Мутация
        else:
            cxpb = 0.4
            mutpb = 0.6

        #Cоздание потомков
        #varAnd применяет кроссовер и мутации к популяции
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        #Оцениваем фитнес только у тех, кто ещё не имеет значений
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)

        #Селекция (NSGA-II), выбираем новое поколение
        population = toolbox.select(population + offspring, k=pop_size)

        if csv_writer:
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            #Логируем лучшее в поколении (по всем целям)
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen] + list(best_ind.fitness.values) + list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    #Возвращаем полный Парето фронт
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values))

    return all_param_names, results, pareto_front

'''
Многокритериальная оптимизация с помощью NSGA-II (Парето-оптимизация) + инициализация начальных параметров
Аргументы:
    target_funcs: список целевых функций
    t_matrix: список массивов времени (оси x ЦФ) для каждой целевой функции
    y_true_matrix: список эталонных данных (ground truth) для каждой целевой функции
    param_bounds: словарь {param_name: (low, high)} для границ параметров. *к сожелению может сделать параметры отрицательными - установить защиту в ЦФ
    loss_func: функция потерь 
    n_gen: число поколений (если None -> определяется автоматически)
    pop_size: размер популяции.
    log_path: путь для логирования в CSV 
    checkpoint_path: путь для сохранения/восстановления состояния
    checkpoint_every: как часто сохранять чекпоинт (в поколениях)
    priorities: веса для каждой ЦФ (штраф на loss)
    init_params: словарь стартовых значений параметров
    delta_init_params: относительный разброс вокруг init_params 
'''
def run_nsga3_optimization_with_init_values(target_funcs, t_matrix, y_true_matrix, param_bounds: dict,
                          loss_func=mse, n_gen=None, pop_size=120,
                          log_path=None, checkpoint_path=None, checkpoint_every=5,
                          priorities=None, init_params=None, delta_init_params=None):
    #Проверки входных данных 
    if len(t_matrix) != len(target_funcs) or len(y_true_matrix) != len(target_funcs):
        raise ValueError(f"Все матрицы должны быть согласованных размеров t_matrix={t_matrix.shape} target_funcs={target_funcs.shape} y_true_matrix={y_true_matrix.shape}")

    #Сбор уникальных имён всех параметров из функций. Механизмы рефлексии
    all_param_names = []
    for f in target_funcs:
        sig = inspect.signature(f)
        f_params = list(sig.parameters.keys())[1:] #первый аргумент всегда t, пропускаем
        all_param_names.extend(f_params)
    all_param_names = list(dict.fromkeys(all_param_names)) #сохраняем порядок, убираем дубли
    
    #Проверяем, что для каждого параметра есть границы
    for name in all_param_names:
        if name not in param_bounds:
            raise ValueError(f"Потеряны границы для '{name}'")
    bounds = [param_bounds[name] for name in all_param_names]

    #Если число поколений не задано — берём минимум 80 или 15 * count_params
    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))
    num_objectives = len(target_funcs)

    #Приоритеты целей 
    if priorities is None:
        priorities = [1.0] * num_objectives
    elif len(priorities) != num_objectives:
        raise ValueError(f"Длина приоритетов ({len(priorities)}) должна соотвестовать количеству ЦФ ({num_objectives})")

    #Создаем типы для NSGA-II (многокритериальная минимизация)

    #Определяем типы индивидов и фитнеса для DEAP 
    #FitnessMulti — это класс фитнеса для многокритериальной задачи.
    #weights=(-1.0,) * num_objectives означает, что все цели минимизируются.
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    #Individual — это класс индивида, связывается с фитнесом FitnessMulti.
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    #Инициализируем toolbox
    toolbox = base.Toolbox()
    #Регистрируем генераторы "генов" и задаем границы
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)

    #Определяем способ создания одного индивида:
    #tools.initCycle вызывает указанные генераторы один раз (n=1) и объединяет их результаты в объект класса Individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    
    #Определяем способ создания популяции:
    #tools.initRepeat повторяет генерацию individual нужное число раз и упаковывает их в обычный список
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Многокритериальная функция оценки 
    '''
    #Без нормализации
    def evaluate(ind):
        losses = []
        for t_i, f, y_true in zip(t_matrix, target_funcs, y_true_matrix):
            sig = inspect.signature(f)
            f_param_names = list(sig.parameters.keys())[1:]
            f_args = [ind[all_param_names.index(p)] for p in f_param_names]
            y_pred = f(t_i, *f_args)
            losses.append(loss_func(y_pred, y_true))
        return tuple(losses)
    '''
    #С нормализацией
    def evaluate(ind):
        losses = []
        for i, (t_i, f, y_true) in enumerate(zip(t_matrix, target_funcs, y_true_matrix)):
            sig = inspect.signature(f)
            f_param_names = list(sig.parameters.keys())[1:]
            f_args = [ind[all_param_names.index(p)] for p in f_param_names]
            y_pred = f(t_i, *f_args)

            #Нормализация
            y_pred = normalize(y_pred)
            y_true = normalize(y_true)

            loss_val = loss_func(y_pred, y_true)
            weighted_loss = loss_val * priorities[i] #Умножаем на приоритет
            losses.append(weighted_loss)
        return tuple(losses)

    toolbox.register("evaluate", evaluate)

    #Операторы генетического алгоритма
    #Комбинирует гены двух родителей и еще раз фиксируем границы
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0) #больше eta - потомки ближе к родителям
    #Мутация генов, фиксация параметров
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2) #вероятность мутации каждого гена
    toolbox.register("select", tools.selNSGA2)

    #Задаём точку отсчёта для hypervolume (чуть выше максимальных значений потерь)
    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    #Восстановление из чекпоинта или инициализация популяции 
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)
        print(f"Checkpoint востановлен на start_gen={start_gen}")
    else:
        if init_params is not None and delta_init_params is not None:
            population = []

            #Создаём индивида по init_params
            exact = creator.Individual([init_params[name] for name in all_param_names])
            population.append(exact)

            #Остальные индивиды — с отклонениями +/- delta
            while len(population) < pop_size:
                indiv_vals = [
                    init_params[name] * (1 + random.uniform(-delta_init_params, delta_init_params))
                    for name in all_param_names
                ]
                population.append(creator.Individual(indiv_vals)) 
        else:
            population = toolbox.population(n=pop_size)

        #Оценка фитнеса и замена плохих на копию exact
        for i in range(len(population)):
            ind = population[i]
            fitness = toolbox.evaluate(ind)
            #Если фитнес некорректный (NaN или Inf), заменяем на init_params
            if any(np.isnan(v) or np.isinf(v) for v in fitness):
                replacement = creator.Individual([init_params[name] for name in all_param_names])
                replacement.fitness.values = toolbox.evaluate(replacement)
                population[i] = replacement
            else:
                ind.fitness.values = fitness

    #Логирование
    if log_path and start_gen == 0:
        log_file = open(log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        header = ['generation'] + [f'tf{i+1}_loss' for i in range(num_objectives)] + all_param_names + ['hypervolume']
        csv_writer.writerow(header)
    elif log_path:
        log_file = open(log_path, mode='a', newline='')
        csv_writer = csv.writer(log_file)
    else:
        csv_writer = None

    prev_hv = 0.0

    #Оптимизация
    for gen in range(start_gen, n_gen):
        #Определяем текущий фронт Парето
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits = [ind.fitness.values for ind in front]

        #Считаем гиперобъём  фронта
        fits_np = np.array(fits)
        current_hv = hv_indicator.do(fits_np)

        #Саморегуляция вероятностей через изменение hypervolume
        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        #Если гиперобъём растёт — уменьшаем мутацию, увеличиваем скрещивание (эксплуатация)
        #Если гиперобъём падает или не растёт — наоборот (исследование)
        if hv_change > 0:
            cxpb = min(0.9, 0.6 + 0.3 * hv_change)  #Скрещивание
            mutpb = max(0.1, 0.4 - 0.3 * hv_change) #Мутация
        else:
            cxpb = 0.4
            mutpb = 0.6

        #Cоздание потомков
        #varAnd применяет кроссовер и мутации к популяции
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        #Оцениваем фитнес только у тех, кто ещё не имеет значений
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)

        #Селекция (NSGA-II), выбираем новое поколение
        population = toolbox.select(population + offspring, k=pop_size)

        if csv_writer:
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            #Логируем лучшее в поколении (по всем целям)
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen] + list(best_ind.fitness.values) + list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    #Возвращаем полный Парето фронт
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values))

    return all_param_names, results, pareto_front



''' !!! Добавить логирование макс и мин отклонения от ЦФ
Многокритериальная оптимизация c помощью NSGA-II (Парето-оптимизация) + инициализация начальных параметров + одна функция для просчета всех ЦФ
Аргументы:
    mega_func: функция для просчета всех ЦФ
    t_matrix: список массивов времени (оси x ЦФ) для каждой целевой функции
    y_true_matrix: список эталонных данных (ground truth) для каждой целевой функции
    param_bounds: словарь {param_name: (low, high)} для границ параметров. *к сожелению может сделать параметры отрицательными - установить защиту в ЦФ
    loss_func: функция потерь 
    n_gen: число поколений (если None -> определяется автоматически)
    pop_size: размер популяции.
    log_path: путь для логирования в CSV 
    checkpoint_path: путь для сохранения/восстановления состояния
    checkpoint_every: как часто сохранять чекпоинт (в поколениях)
    priorities: веса для каждой ЦФ (штраф на loss)
    init_params: словарь стартовых значений параметров
    delta_init_params: относительный разброс вокруг init_params 
'''
def run_nsga3_optimization_with_init_values2_min_max_logging(
        mega_func, t_matrix, y_true_matrix, param_bounds: dict,
        loss_func=mse, n_gen=None, pop_size=120,
        log_path=None, checkpoint_path=None, checkpoint_every=5,
        priorities=None, init_params=None, delta_init_params=None):

    # Проверки входных данных
    num_objectives = len(y_true_matrix)
    if len(t_matrix) != num_objectives:
        raise ValueError(
            f"t_matrix и y_true_matrix должны иметь одинаковое число функций: "
            f"len(t_matrix)={len(t_matrix)}, len(y_true_matrix)={len(y_true_matrix)}"
        )

    # Сбор имён параметров напрямую из словаря param_bounds
    all_param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in all_param_names]

    # Если число поколений не задано — берём минимум 80 или 15 * count_params
    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))

    # Приоритеты целей
    if priorities is None:
        priorities = [1.0] * num_objectives
    elif len(priorities) != num_objectives:
        raise ValueError(
            f"Длина приоритетов ({len(priorities)}) должна соответствовать количеству ЦФ ({num_objectives})"
        )

    # Определяем типы индивидов и фитнеса для DEAP
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Инициализируем toolbox
    toolbox = base.Toolbox()
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Многокритериальная функция оценки (через mega_func)
    def evaluate(ind):
        f_args = [ind[all_param_names.index(p)] for p in all_param_names]
        y_pred_matrix = mega_func(t_matrix, *f_args)

        losses = []
        deviations = []  # для хранения min/max отклонений
        for i, y_true in enumerate(y_true_matrix):
            y_pred = y_pred_matrix[i]
            y_pred = normalize(y_pred)
            y_true = normalize(y_true)

            loss_val = loss_func(y_pred, y_true)
            weighted_loss = loss_val * priorities[i]
            losses.append(weighted_loss)

            # вычисляем отклонения
            diff = y_pred - y_true
            deviations.append((float(np.min(diff)), float(np.max(diff))))

        # сохраняем отклонения в атрибут индивида для логирования
        ind.deviations = deviations
        return tuple(losses)

    toolbox.register("evaluate", evaluate)

    # Операторы генетического алгоритма
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    # Восстановление из чекпоинта или инициализация популяции
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)
        print(f"Checkpoint восстановлен на start_gen={start_gen}")
    else:
        if init_params is not None and delta_init_params is not None:
            population = []
            exact = creator.Individual([init_params[name] for name in all_param_names])
            population.append(exact)

            while len(population) < pop_size:
                indiv_vals = [
                    init_params[name] * (1 + random.uniform(-delta_init_params, delta_init_params))
                    for name in all_param_names
                ]
                population.append(creator.Individual(indiv_vals))
        else:
            population = toolbox.population(n=pop_size)

        for i in range(len(population)):
            ind = population[i]
            fitness = toolbox.evaluate(ind)
            if any(np.isnan(v) or np.isinf(v) for v in fitness):
                replacement = creator.Individual([init_params[name] for name in all_param_names])
                replacement.fitness.values = toolbox.evaluate(replacement)
                population[i] = replacement
            else:
                ind.fitness.values = fitness

    # Логирование
    if log_path and start_gen == 0:
        log_file = open(log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        header = ['generation']
        for i in range(num_objectives):
            header += [f'tf{i+1}_loss', f'tf{i+1}_min_dev', f'tf{i+1}_max_dev']
        header += all_param_names + ['hypervolume']
        csv_writer.writerow(header)
    elif log_path:
        log_file = open(log_path, mode='a', newline='')
        csv_writer = csv.writer(log_file)
    else:
        csv_writer = None

    prev_hv = 0.0

    # Оптимизация
    for gen in range(start_gen, n_gen):
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits = [ind.fitness.values for ind in front]

        fits_np = np.array(fits)
        current_hv = hv_indicator.do(fits_np)

        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        if hv_change > 0:
            cxpb = min(0.9, 0.6 + 0.3 * hv_change)
            mutpb = max(0.1, 0.4 - 0.3 * hv_change)
        else:
            cxpb = 0.4
            mutpb = 0.6

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)

        population = toolbox.select(population + offspring, k=pop_size)

        if csv_writer:
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen]
                for loss, (min_dev, max_dev) in zip(best_ind.fitness.values, best_ind.deviations):
                    row += [loss, min_dev, max_dev]
                row += list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    # Возвращаем полный Парето фронт
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values, getattr(ind, 'deviations', None)))

    return all_param_names, results, pareto_front

def run_nsga3_optimization_with_init_values2(
        mega_func, t_matrix, y_true_matrix, param_bounds: dict,
        loss_func=mse, n_gen=None, pop_size=120,
        log_path=None, checkpoint_path=None, checkpoint_every=5,
        priorities=None, init_params=None, delta_init_params=None):

    # Проверки входных данных
    num_objectives = len(y_true_matrix)
    if len(t_matrix) != num_objectives:
        raise ValueError(
            f"t_matrix и y_true_matrix должны иметь одинаковое число функций: "
            f"len(t_matrix)={len(t_matrix)}, len(y_true_matrix)={len(y_true_matrix)}"
        )

    # Сбор имён параметров напрямую из словаря param_bounds
    all_param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in all_param_names]

    # Если число поколений не задано — берём минимум 80 или 15 * count_params
    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))

    # Приоритеты целей
    if priorities is None:
        priorities = [1.0] * num_objectives
    elif len(priorities) != num_objectives:
        raise ValueError(
            f"Длина приоритетов ({len(priorities)}) должна соответствовать количеству ЦФ ({num_objectives})"
        )

    # Определяем типы индивидов и фитнеса для DEAP
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Инициализируем toolbox
    toolbox = base.Toolbox()
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Многокритериальная функция оценки (через mega_func)
    def evaluate(ind):
        f_args = [ind[all_param_names.index(p)] for p in all_param_names]
        # mega_func возвращает матрицу num_objectives x len(t)
        y_pred_matrix = mega_func(t_matrix, *f_args)

        losses = []
        for i, y_true in enumerate(y_true_matrix):
            y_pred = y_pred_matrix[i]
            y_pred = normalize(y_pred)
            y_true = normalize(y_true)

            loss_val = loss_func(y_pred, y_true)
            weighted_loss = loss_val * priorities[i]
            losses.append(weighted_loss)
        return tuple(losses)

    toolbox.register("evaluate", evaluate)

    # Операторы генетического алгоритма
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0)
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    # Восстановление из чекпоинта или инициализация популяции
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)
        print(f"Checkpoint восстановлен на start_gen={start_gen}")
    else:
        if init_params is not None and delta_init_params is not None:
            population = []
            exact = creator.Individual([init_params[name] for name in all_param_names])
            population.append(exact)

            while len(population) < pop_size:
                indiv_vals = [
                    init_params[name] * (1 + random.uniform(-delta_init_params, delta_init_params))
                    for name in all_param_names
                ]
                population.append(creator.Individual(indiv_vals))
        else:
            population = toolbox.population(n=pop_size)

        for i in range(len(population)):
            ind = population[i]
            fitness = toolbox.evaluate(ind)
            if any(np.isnan(v) or np.isinf(v) for v in fitness):
                replacement = creator.Individual([init_params[name] for name in all_param_names])
                replacement.fitness.values = toolbox.evaluate(replacement)
                population[i] = replacement
            else:
                ind.fitness.values = fitness

    # Логирование
    if log_path and start_gen == 0:
        log_file = open(log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        header = ['generation'] + [f'tf{i+1}_loss' for i in range(num_objectives)] + all_param_names + ['hypervolume']
        csv_writer.writerow(header)
    elif log_path:
        log_file = open(log_path, mode='a', newline='')
        csv_writer = csv.writer(log_file)
    else:
        csv_writer = None

    prev_hv = 0.0

    # Оптимизация
    for gen in range(start_gen, n_gen):
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits = [ind.fitness.values for ind in front]

        fits_np = np.array(fits)
        current_hv = hv_indicator.do(fits_np)

        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        if hv_change > 0:
            cxpb = min(0.9, 0.6 + 0.3 * hv_change)
            mutpb = max(0.1, 0.4 - 0.3 * hv_change)
        else:
            cxpb = 0.4
            mutpb = 0.6

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)

        population = toolbox.select(population + offspring, k=pop_size)

        if csv_writer:
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen] + list(best_ind.fitness.values) + list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    # Возвращаем полный Парето фронт
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values))

    return all_param_names, results, pareto_front


def run_nsga3_optimization_with_init_values3(target_funcs, t_matrix, y_true_matrix, param_bounds: dict,
                          loss_func=mse, n_gen=None, pop_size=120,
                          log_path=None, checkpoint_path=None, checkpoint_every=5,
                          priorities=None, init_params=None, delta_init_params=None):
    #Проверки входных данных 
    #Ожидаем одну Мега-ЦФ (callable)
    if not callable(target_funcs):
        raise ValueError("Ожидается одна Мега-ЦФ (callable), а не список функций.")
    if len(t_matrix) != len(y_true_matrix):
        raise ValueError(f"t_matrix и y_true_matrix должны иметь одинаковую длину; len(t_matrix)={len(t_matrix)}, len(y_true_matrix)={len(y_true_matrix)}")

    #Сбор имён параметров из сигнатуры единственной Мега-ЦФ (пропускаем первый аргумент t_matrix)
    sig = inspect.signature(target_funcs)
    all_param_names = list(sig.parameters.keys())[1:]
    
    #Проверяем, что для каждого параметра есть границы
    for name in all_param_names:
        if name not in param_bounds:
            raise ValueError(f"Потеряны границы для '{name}'")
    bounds = [param_bounds[name] for name in all_param_names]

    #Если число поколений не задано — берём минимум 80 или 15 * count_params
    if n_gen is None:
        n_gen = max(80, 15 * len(bounds))
    
    #Определяем число целей по первому y_true (поддерживаем (num_obj, len_t) или (len_t, num_obj) или 1D)
    sample_y = np.asarray(y_true_matrix[0])
    if sample_y.ndim == 2:
        #Если одна размерность явно равна числу целей — берём максимум по меньшей логике
        if sample_y.shape[0] >= sample_y.shape[1]:
            num_objectives = sample_y.shape[0]
        else:
            num_objectives = sample_y.shape[1]
    elif sample_y.ndim == 1:
        num_objectives = 1
    else:
        raise ValueError(f"Неподдерживаемая форма y_true_matrix[0]: {sample_y.shape}")

    #Приоритеты целей 
    if priorities is None:
        priorities = [1.0] * num_objectives

    #Создаем типы для NSGA-II (многокритериальная минимизация)

    #Определяем типы индивидов и фитнеса для DEAP 
    #FitnessMulti — это класс фитнеса для многокритериальной задачи.
    #weights=(-1.0,) * num_objectives означает, что все цели минимизируются.
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * num_objectives)
    #Individual — это класс индивида, связывается с фитнесом FitnessMulti.
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    #Инициализируем toolbox
    toolbox = base.Toolbox()
    #Регистрируем генераторы "генов" и задаем границы
    for i, (lo, hi) in enumerate(bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)

    #Определяем способ создания одного индивида:
    #tools.initCycle вызывает указанные генераторы один раз (n=1) и объединяет их результаты в объект класса Individual
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     [toolbox.__getattribute__(f"attr_{i}") for i in range(len(bounds))], n=1)
    
    #Определяем способ создания популяции:
    #tools.initRepeat повторяет генерацию individual нужное число раз и упаковывает их в обычный список
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #Многокритериальная функция оценки 
    '''
    def evaluate(ind):
        #Параметры индивида
        params = [ind[all_param_names.index(p)] for p in all_param_names]

        #Вызов мега-ЦФ (ожидается матрица: n_obj * len_t)
        y_pred_matrix = np.asarray(target_funcs(t_matrix, *params), dtype=float)
        y_true_matrix_arr = np.asarray(y_true_matrix, dtype=float)

        #Приведение формы
        if y_pred_matrix.shape != y_true_matrix_arr.shape:
            min_shape = tuple(min(a, b) for a, b in zip(y_pred_matrix.shape, y_true_matrix_arr.shape))
            y_pred_matrix = y_pred_matrix[:min_shape[0], :min_shape[1]]
            y_true_matrix_arr = y_true_matrix_arr[:min_shape[0], :min_shape[1]]

        #Вычисляем loss по каждой строке 
        losses = []
        for i in range(y_true_matrix_arr.shape[0]):
            y_pred = normalize(y_pred_matrix[i])
            y_true = normalize(y_true_matrix_arr[i])
            loss_val = float(loss_func(y_pred, y_true))
            weighted_loss = loss_val * priorities[i] 
            losses.append(weighted_loss)

        return tuple(losses)
     '''
    
    def evaluate(ind):
        #Параметры индивида
        params = [ind[all_param_names.index(p)] for p in all_param_names]

        #Вызов мега-ЦФ (ожидается список массивов для каждой цели)
        y_pred_list = target_funcs(t_matrix, *params)  # список, длины могут быть разными

        print(f'y_pred_list: {y_pred_list}')

        #Вычисляем loss по каждой цели
        losses = []
        for i, y_pred_i in enumerate(y_pred_list):
            print(f'y_true_matrix[i]: {y_true_matrix[i]}')
            print(f'y_pred_i: {y_pred_i}')
            '''
            y_pred = normalize(np.asarray(y_pred_i, dtype=float))
            ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. 
            The detected shape was (6,) + inhomogeneous part.  
            '''
            y_pred = normalize(np.asarray(y_pred_i, dtype=float))
            y_true = normalize(np.asarray(y_true_matrix[i], dtype=float))
            loss_val = float(loss_func(y_pred, y_true))
            weighted_loss = loss_val * priorities[i]
            losses.append(weighted_loss)

        return tuple(losses)

    toolbox.register("evaluate", evaluate)

    #Операторы генетического алгоритма
    #Комбинирует гены двух родителей и еще раз фиксируем границы
    toolbox.register("mate", tools.cxSimulatedBinaryBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=15.0) #больше eta - потомки ближе к родителям
    #Мутация генов, фиксация параметров
    toolbox.register("mutate", tools.mutPolynomialBounded,
                     low=[lo for lo, hi in bounds],
                     up=[hi for lo, hi in bounds],
                     eta=20.0,
                     indpb=0.2) #вероятность мутации каждого гена
    toolbox.register("select", tools.selNSGA2)

    #Задаём точку отсчёта для hypervolume (чуть выше максимальных значений потерь)
    reference_point = [1.2] * num_objectives
    hv_indicator = HV(ref_point=reference_point)

    #Восстановление из чекпоинта или инициализация популяции 
    start_gen = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        start_gen = checkpoint["generation"] + 1
        population = [creator.Individual(ind) for ind in checkpoint["population"]]
        for ind, fit in zip(population, checkpoint["fitnesses"]):
            ind.fitness.values = tuple(fit)
        print(f"Checkpoint востановлен на start_gen={start_gen}")
    else:
        if init_params is not None and delta_init_params is not None:
            population = []

            #Создаём индивида по init_params
            exact = creator.Individual([init_params[name] for name in all_param_names])
            population.append(exact)

            #Остальные индивиды — с отклонениями +/- delta
            while len(population) < pop_size:
                indiv_vals = [
                    init_params[name] * (1 + random.uniform(-delta_init_params, delta_init_params))
                    for name in all_param_names
                ]
                population.append(creator.Individual(indiv_vals)) 
        else:
            population = toolbox.population(n=pop_size)

        #Оценка фитнеса и замена плохих на копию exact
        for i in range(len(population)):
            ind = population[i]
            fitness = toolbox.evaluate(ind)
            #Если фитнес некорректный (NaN или Inf), заменяем на init_params
            if any(np.isnan(v) or np.isinf(v) for v in fitness):
                replacement = creator.Individual([init_params[name] for name in all_param_names])
                replacement.fitness.values = toolbox.evaluate(replacement)
                population[i] = replacement
            else:
                ind.fitness.values = fitness

    #Логирование
    if log_path and start_gen == 0:
        log_file = open(log_path, mode='w', newline='')
        csv_writer = csv.writer(log_file)
        header = ['generation'] + [f'tf{i+1}_loss' for i in range(num_objectives)] + all_param_names + ['hypervolume']
        csv_writer.writerow(header)
    elif log_path:
        log_file = open(log_path, mode='a', newline='')
        csv_writer = csv.writer(log_file)
    else:
        csv_writer = None

    prev_hv = 0.0

    #Оптимизация
    for gen in range(start_gen, n_gen):
        #Определяем текущий фронт Парето
        front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
        fits = [ind.fitness.values for ind in front]

        #Считаем гиперобъём  фронта
        fits_np = np.array(fits)
        current_hv = hv_indicator.do(fits_np)

        #Саморегуляция вероятностей через изменение hypervolume
        hv_change = current_hv - prev_hv
        prev_hv = current_hv

        #Если гиперобъём растёт — уменьшаем мутацию, увеличиваем скрещивание (эксплуатация)
        #Если гиперобъём падает или не растёт — наоборот (исследование)
        if hv_change > 0:
            cxpb = min(0.9, 0.6 + 0.3 * hv_change)  #Скрещивание
            mutpb = max(0.1, 0.4 - 0.3 * hv_change) #Мутация
        else:
            cxpb = 0.4
            mutpb = 0.6

        #Cоздание потомков
        #varAnd применяет кроссовер и мутации к популяции
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        #Оцениваем фитнес только у тех, кто ещё не имеет значений
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_inds:
            ind.fitness.values = toolbox.evaluate(ind)

        #Селекция (NSGA-II), выбираем новое поколение
        population = toolbox.select(population + offspring, k=pop_size)

        if csv_writer:
            for ind in population:
                if not ind.fitness.valid:
                    ind.fitness.values = toolbox.evaluate(ind)
            #Логируем лучшее в поколении (по всем целям)
            best_inds = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
            for best_ind in best_inds:
                row = [gen] + list(best_ind.fitness.values) + list(best_ind) + [current_hv]
                csv_writer.writerow(row)

        if checkpoint_path and checkpoint_every and (gen % checkpoint_every == 0):
            checkpoint_data = {
                "generation": gen,
                "population": [list(ind) for ind in population],
                "fitnesses": [list(ind.fitness.values) for ind in population],
                "param_names": all_param_names,
            }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_data, f)

    if csv_writer:
        log_file.close()

    #Возвращаем полный Парето фронт
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    results = []
    for ind in pareto_front:
        clamped = [max(lo, min(ind[i], hi)) for i, (lo, hi) in enumerate(bounds)]
        results.append((clamped, ind.fitness.values))

    return all_param_names, results, pareto_front
