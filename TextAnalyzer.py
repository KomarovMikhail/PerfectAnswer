import pymorphy2
import numpy as np
from scipy.spatial import distance
import pandas as pd


class TextAnalyser:
    def __init__(self, input_path, output_path, perfect_answer, stops=None, symbols=''):

        self._perfect_answer = ''
        self._stops = stops
        self._symbols = symbols
        self._output = output_path
        self._perfect_answer_path = perfect_answer

        df = pd.read_excel(input_path, names=['id', 'answer'])
        self._answers = df.values

    def _split_string(self, string):
        """
        Приведение строки к нижнему регистру и разбиение ее на слова
        :param string: строка
        :return: массив слов исходной строки в нижнем регистре
        """
        if type(string) is not str:
            return []

        s = string.lower()
        chars = list(s)
        for i in range(len(chars)):
            if chars[i] in self._symbols:
                chars[i] = ' '
        result = ''.join(chars)
        return result.split()

    def _get_stops(self):
        """
        Функция возвращает список стоп слов
        """
        if self._stops is None:
            return []
        with open(self._stops, 'r') as file:
            stops = file.read().split('\n')
        return stops

    @staticmethod
    def _compress_table(table):
        """
        SVD разложение с исключением мало значимых тем
        :param table: tf-idf матрица слова/ответы
        :return: Матрцицу с выделенными основными темами и учитывающую скрытые зависимости
        """
        u, s, vh = np.linalg.svd(table, full_matrices=False)

        k = 0
        for item in s:
            if item < 1e-5:
                break
            k += 1

        u = u[:, 0:k]
        s = s[0:k]
        vh = vh[0:k, :]
        result = np.dot(u, np.dot(np.diag(s), vh))

        return result

    @staticmethod
    def _set_completeness(completeness,  i, string):
        res = 0.0
        risc_count = string.count('риск')
        sol_count = string.count('мера')
        if risc_count > 2:
            res += 3
        else:
            res += risc_count
        if sol_count > 2:
            res += 3
        else:
            res += sol_count
        completeness[i] = res / 6

    @staticmethod
    def _set_structure(structure, i, string):
        res = 0.0
        if '1' in string:
            res += 1
        if '2' in string:
            res += 1
        if '3' in string:
            res += 1
        structure[i] = res / 3

    def _set_distance(self, table):
        result = []
        for i in range(len(self._answers)):
            result.append(np.sum(table[0] * table[i + 1]))
        return result

    def _preprocess(self):
        """
        Предварительная обработка данных: приводит слова к нормальной форма, удаляет все стоп-слова.
        Выссчитывает значение коэффициентов completeness и structure для каждого из ответов
        :return: три списка:
        1) значения коэффициентов completeness для каждого из ответов
        2) значения коэффициентов structure для каждого из ответов
        3) количество слов в каждом из ответов
        """
        analyzer = pymorphy2.MorphAnalyzer()
        stops = self._get_stops()

        with open(self._perfect_answer_path, 'r') as file:
            perfect_answer = file.read()
        perfect_answer = self._split_string(perfect_answer)
        perfect_answer = [analyzer.parse(word)[0].normal_form for word in perfect_answer if word not in stops]
        self._perfect_answer = perfect_answer

        answers_num = len(self._answers)
        structure = np.zeros(answers_num)
        completeness = np.zeros(answers_num)
        words_num = np.zeros(answers_num + 1)

        words_num[0] = len(perfect_answer)

        for i in range(answers_num):
            new_answer = self._split_string(self._answers[i][1])
            new_answer = [analyzer.parse(word)[0].normal_form for word in new_answer if word not in stops]

            self._answers[i][1] = new_answer

            words_num[i + 1] = len(new_answer)
            self._set_completeness(completeness, i, new_answer)
            self._set_structure(structure, i, new_answer)

        return completeness, structure, words_num

    def _get_table(self, words_num):
        """
        Считает tf-idf матрицу для заданных ответов, при этом исключая слова,
        которые встречаются только один раз.
        :param words_num: массив количества слов в каждом документе
        :return: Двумерную матрицу размерности (количество различных слов во всех ответах, количество ответов)
        """

        table = {}

        # проходимся по всем словам, составляем частотную матрицу для всех слов/ответов
        for i in range(len(self._answers)):
            for word in self._answers[i][1]:
                if table.get(word) is None:
                    table[word] = np.zeros(len(self._answers) + 1)
                table[word][i + 1] += 1

        for word in self._perfect_answer:
            if table.get(word) is None:
                table[word] = np.zeros(len(self._answers) + 1)
            table[word][0] += 1

        result = np.array([list(item) for item in table.values()])

        # Вычисляем tf-idf и заодно помечаем слова, которые встречаются только один раз
        indexes_to_delete = []
        for i in range(len(result)):
            if sum(result[i]) == 1:
                indexes_to_delete.append(i)
            else:
                result[i] /= words_num * np.log10(result.shape[1] / np.nansum(result[i]/result[i]))

        # удаляем слова которые встречаются только один раз из матрицы
        return np.delete(result, tuple(indexes_to_delete), axis=0)

    @staticmethod
    def _calc_cosine_distance(table):
        """
        Считает косинусное расстояние между идеальным ответом и всеми остальными
        :param table: матрица, полученная в результате svd- разложения
        :return: список косинусных расстояний для каждого из ответов
        """
        i = 1
        result = []
        while i < table.shape[1]:
            dist = distance.cosine(table[:, 0], table[:, i])
            if np.isnan(dist) or 1 - dist < 0:
                result.append(0)
            else:
                result.append(1 - dist)
            i += 1
        return np.array(result)

    def _write_results(self, measure):
        ids = [self._answers[i][0] for i in range(len(self._answers))]

        d = {'ID ответа': ids, 'Степень сходства с идеальным ответом': measure}
        df = pd.DataFrame(data=d)
        df.to_excel(self._output)

    @staticmethod
    def _get_measure(completeness, structure, dist):
        return (completeness * 0.5 + structure * 0.5 + dist * 2) / 3

    def run(self):
        completeness, structure, words_num = self._preprocess()
        table = np.nan_to_num(self._get_table(np.array(words_num)))

        table = self._compress_table(table)

        dist = TextAnalyser._calc_cosine_distance(table)

        self._write_results(TextAnalyser._get_measure(completeness, structure, dist))


