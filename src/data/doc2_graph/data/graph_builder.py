import json
import os
from PIL import Image, ImageDraw
from typing import Tuple
import torch
import dgl
import random
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET

from .preprocessing import load_predictions
from .utils import polar
from ..paths import DATA, FUNSD_TEST
from ..utils import get_config


class GraphBuilder():

    def __init__(self):
        self.cfg_preprocessing = get_config('preprocessing')
        self.edge_type = self.cfg_preprocessing.GRAPHS.edge_type
        self.data_type = self.cfg_preprocessing.GRAPHS.data_type
        self.node_granularity = self.cfg_preprocessing.GRAPHS.node_granularity
        random.seed = 42
        return

    def get_graph(self, src_path : str, src_data : str) -> Tuple[list, list, list, list]:
        """ Given the source, it returns a graph

        Args:
            src_path (str) : path to source data
            src_data (str) : either FUNSD, PAU or CUSTOM

        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """

        if src_data == 'FUNSD':
            return self.__fromFUNSD(src_path)
        elif src_data == 'PAU':
            return self.__fromPAU(src_path)
        elif src_data == 'CUSTOM':
            if self.data_type == 'img':
                return self.__fromIMG()
            elif self.data_type == 'pdf':
                return self.__fromPDF()
            else:
                raise Exception('GraphBuilder exception: data type invalid. Choose from ["img", "pdf"]')
        else:
            raise Exception('GraphBuilder exception: source data invalid. Choose from ["FUNSD", "PAU", "CUSTOM"]')

    def balance_edges(self, g : dgl.DGLGraph, cls=None ) -> dgl.DGLGraph:
        """ if cls (class) is not None, but an integer instead, balance that class to be equal to the sum of the other classes

        Args:
            g (DGLGraph) : a DGL graph
            cls (int) : class number, if any

        Returns:
            g (DGLGraph) : the new balanced graph
        """

        edge_targets = g.edata['label']
        u, v = g.all_edges(form='uv')
        edges_list = list()
        for e in zip(u.tolist(), v.tolist()):
            edges_list.append([e[0], e[1]])

        if type(cls) is int:
            to_remove = (edge_targets == cls)
            indices_to_remove = to_remove.nonzero().flatten().tolist()

            for _ in range(int((edge_targets != cls).sum()/2)):
                indeces_to_save = [random.choice(indices_to_remove)]
                edge = edges_list[indeces_to_save[0]]

                for index in sorted(indeces_to_save, reverse=True):
                    del indices_to_remove[indices_to_remove.index(index)]

            indices_to_remove = torch.flatten(torch.tensor(indices_to_remove, dtype=torch.int32))
            g = dgl.remove_edges(g, indices_to_remove)
            return g

        else:
            raise Exception("Select a class to balance (an integer ranging from 0 to num_edge_classes).")

    def get_info(self):
        """ returns graph information
        """
        print(f"-> edge type: {self.edge_type}")

    def fully_connected(self, ids : list) -> Tuple[list, list]:
        """ create fully connected graph

        Args:
            ids (list) : list of node indices

        Returns:
            u, v (lists) : lists of indices
        """
        u, v = list(), list()
        for id in ids:
            u.extend([id for i in range(len(ids)) if i != id])
            v.extend([i for i in range(len(ids)) if i != id])
        return u, v

    def __knn(self, size : tuple, bboxs : list, k = 10) -> Tuple[list, list]:
        """ Given a list of bounding boxes, find for each of them their k nearest ones.

        Args:
            size (tuple) : width and height of the image
            bboxs (list) : list of bounding box coordinates
            k (int) : k of the knn algorithm

        Returns:
            u, v (lists) : lists of indices
        """

        edges = []
        width, height = size[0], size[1]

        # creating projections
        vertical_projections = [[] for i in range(width)]
        horizontal_projections = [[] for i in range(height)]
        for node_index, bbox in enumerate(bboxs):
            for hp in range(bbox[0], bbox[2]):
                if hp >= width: hp = width - 1
                vertical_projections[hp].append(node_index)
            for vp in range(bbox[1], bbox[3]):
                if vp >= height: vp = height - 1
                horizontal_projections[vp].append(node_index)

        def bound(a, ori=''):
            if a < 0 : return 0
            elif ori == 'h' and a > height: return height
            elif ori == 'w' and a > width: return width
            else: return a

        for node_index, node_bbox in enumerate(bboxs):
            neighbors = [] # collect list of neighbors
            window_multiplier = 2 # how much to look around bbox
            wider = (node_bbox[2] - node_bbox[0]) > (node_bbox[3] - node_bbox[1]) # if bbox wider than taller

            ### finding neighbors ###
            while(len(neighbors) < k and window_multiplier < 100): # keep enlarging the window until at least k bboxs are found or window too big
                vertical_bboxs = []
                horizontal_bboxs = []
                neighbors = []

                if wider:
                    h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier/4)
                    v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier)
                else:
                    h_offset = int((node_bbox[2] - node_bbox[0]) * window_multiplier)
                    v_offset = int((node_bbox[3] - node_bbox[1]) * window_multiplier/4)

                window = [bound(node_bbox[0] - h_offset),
                        bound(node_bbox[1] - v_offset),
                        bound(node_bbox[2] + h_offset, 'w'),
                        bound(node_bbox[3] + v_offset, 'h')]

                [vertical_bboxs.extend(d) for d in vertical_projections[window[0]:window[2]]]
                [horizontal_bboxs.extend(d) for d in horizontal_projections[window[1]:window[3]]]

                for v in set(vertical_bboxs):
                    for h in set(horizontal_bboxs):
                        if v == h: neighbors.append(v)

                window_multiplier += 1 # enlarge the window

            ### finding k nearest neighbors ###
            neighbors = list(set(neighbors))
            if node_index in neighbors:
                neighbors.remove(node_index)
            neighbors_distances = [polar(node_bbox, bboxs[n])[0] for n in neighbors]
            for sd_num, sd_idx in enumerate(np.argsort(neighbors_distances)):
                if sd_num < k:
                    if [node_index, neighbors[sd_idx]] not in edges and [neighbors[sd_idx], node_index] not in edges:
                        edges.append([neighbors[sd_idx], node_index])
                        edges.append([node_index, neighbors[sd_idx]])
                else: break

        return [e[0] for e in edges], [e[1] for e in edges]

    def _discrete_position(self, rect_src : list, rect_dst : list) -> int:
        """ Compute discrete direction from src to dst (9 bins) """
        left = (rect_dst[2] - rect_src[0]) <= 0
        bottom = (rect_src[3] - rect_dst[1]) <= 0
        right = (rect_src[2] - rect_dst[0]) <= 0
        top = (rect_dst[3] - rect_src[1]) <= 0

        vp_intersect = (rect_src[0] <= rect_dst[2] and rect_dst[0] <= rect_src[2])
        hp_intersect = (rect_src[1] <= rect_dst[3] and rect_dst[1] <= rect_src[3])
        rect_intersect = vp_intersect and hp_intersect

        if rect_intersect: return 0
        elif top and left: return 1
        elif left and bottom: return 2
        elif bottom and right: return 3
        elif right and top: return 4
        elif left: return 5
        elif right: return 6
        elif bottom: return 7
        elif top: return 8
        return 0

    def _score_candidate_edge(self, box_i, box_j, page_w, page_h, pool_max_dist, weights, texts_i=None, texts_j=None):
        """ Score a candidate edge (i, j) using multiple layout-aware signals.

        Returns: (score, signal_dict)
        """
        signals = {}

        center = lambda rect: ((rect[2]+rect[0])/2.0, (rect[3]+rect[1])/2.0)
        cx_i, cy_i = center(box_i)
        cx_j, cy_j = center(box_j)

        # Distance signal: 1 - (dist / max_dist)
        dist_euclidean = ((cx_i - cx_j)**2 + (cy_i - cy_j)**2) ** 0.5
        s_dist = max(0, 1.0 - (dist_euclidean / max(pool_max_dist, 1)))
        signals['distance'] = s_dist

        # Row alignment (horizontal): 1 - (|cy_i - cy_j| / page_h)
        s_row = max(0, 1.0 - (abs(cy_i - cy_j) / max(page_h, 1)))
        signals['row_align'] = s_row

        # Column alignment (vertical): 1 - (|cx_i - cx_j| / page_w)
        s_col = max(0, 1.0 - (abs(cx_i - cx_j) / max(page_w, 1)))
        signals['col_align'] = s_col

        # Direction signal: bonus for natural reading order (right, below)
        direction_bonuses = [0.5, 0.1, 0.1, 0.7, 0.4, 0.3, 1.0, 1.0, 0.3]
        direction_idx = self._discrete_position(box_i, box_j)
        s_dir = direction_bonuses[direction_idx]
        signals['direction'] = s_dir

        # Overlap signal: IoU or projected overlap
        def iou(b1, b2):
            xi1 = max(b1[0], b2[0])
            yi1 = max(b1[1], b2[1])
            xi2 = min(b1[2], b2[2])
            yi2 = min(b1[3], b2[3])
            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            u1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            u2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = u1 + u2 - inter
            return inter / max(union, 1)

        def proj_overlap(b1, b2, axis='h'):
            if axis == 'h':
                overlap = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                extent = min(b1[3] - b1[1], b2[3] - b2[1])
            else:
                overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
                extent = min(b1[2] - b1[0], b2[2] - b2[0])
            return overlap / max(extent, 1)

        s_overlap = max(iou(box_i, box_j), proj_overlap(box_i, box_j, 'h'), proj_overlap(box_i, box_j, 'v'))
        signals['overlap'] = s_overlap

        # Text similarity (if texts provided, else 0)
        s_text = 0.0
        if texts_i and texts_j:
            tokens_i = set(texts_i.lower().split())
            tokens_j = set(texts_j.lower().split())
            intersection = len(tokens_i & tokens_j)
            union = len(tokens_i | tokens_j)
            s_text = intersection / max(union, 1)
        signals['text_sim'] = s_text

        # Combine signals
        score = 0.0
        for signal_name, weight in weights.items():
            if weight > 0 and signal_name in signals:
                score += weight * signals[signal_name]

        return score, signals

    def __adaptive(self, size : tuple, bboxs : list, texts : list = None) -> Tuple[list, list, dict]:
        """ Build edges using adaptive scoring of candidate neighbors.

        Args:
            size (tuple) : (width, height) of page
            bboxs (list) : list of bbox coordinates [x1, y1, x2, y2]
            texts (list) : optional text for each bbox

        Returns:
            u, v (lists) : edge source/dest indices
            edge_signals (dict) : per-edge signal values for debugging/ablation
        """

        cfg_adaptive = self.cfg_preprocessing.ADAPTIVE
        pool_size = cfg_adaptive.candidate_pool_size
        adaptive_k = cfg_adaptive.adaptive_k
        threshold = cfg_adaptive.score_threshold
        weights = cfg_adaptive.weights

        width, height = size[0], size[1]
        edges = []
        edge_signals_list = []

        for node_i in range(len(bboxs)):
            # Step 1: compute distances to all other nodes
            distances = []
            for node_j in range(len(bboxs)):
                if node_i == node_j:
                    distances.append(float('inf'))
                else:
                    c_i = ((bboxs[node_i][2] + bboxs[node_i][0]) / 2.0, (bboxs[node_i][3] + bboxs[node_i][1]) / 2.0)
                    c_j = ((bboxs[node_j][2] + bboxs[node_j][0]) / 2.0, (bboxs[node_j][3] + bboxs[node_j][1]) / 2.0)
                    dist = ((c_i[0] - c_j[0])**2 + (c_i[1] - c_j[1])**2) ** 0.5
                    distances.append(dist)

            # Step 2: get top pool_size candidates by distance
            candidate_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:pool_size]
            pool_max_dist = distances[candidate_indices[-1]] if candidate_indices else 1

            # Step 3: score each candidate
            candidate_scores = []
            for node_j in candidate_indices:
                score, signals = self._score_candidate_edge(
                    bboxs[node_i], bboxs[node_j], width, height, pool_max_dist,
                    weights,
                    texts[node_i] if texts else None,
                    texts[node_j] if texts else None
                )
                candidate_scores.append((node_j, score, signals))

            # Step 4: select top-k by score, apply threshold
            candidate_scores.sort(key=lambda x: x[1], reverse=True)
            selected = [(node_j, score, sig) for node_j, score, sig in candidate_scores
                       if score >= threshold][:adaptive_k]

            # Step 5: emit bidirectional edges
            for node_j, score, signals in selected:
                edge_pair = (node_i, node_j) if node_i < node_j else (node_j, node_i)
                if edge_pair not in edges and (edge_pair[1], edge_pair[0]) not in edges:
                    edges.append((node_i, node_j))
                    edges.append((node_j, node_i))
                    edge_signals_list.append(signals)
                    edge_signals_list.append(signals)

        return [e[0] for e in edges], [e[1] for e in edges]

    def __fromIMG():
        #TODO: dev from IMG import of graphs
        return

    def __fromPDF():
        #TODO: dev from PDF import of graphs
        return

    def __fromPAU(self, src: str) -> Tuple[list, list, list, list]:
        """ build graphs from Pau Riba's dataset

        Args:
            src (str) : path to where data is stored

        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': [], 'texts': [], 'boxs': []}

        for image in tqdm(os.listdir(src), desc='Creating graphs'):
            if not image.endswith('tif'): continue

            img_name = image.split('.')[0]
            file_gt = img_name + '_gt.xml'
            file_ocr = img_name + '_ocr.xml'

            if not os.path.isfile(os.path.join(src, file_gt)) or not os.path.isfile(os.path.join(src, file_ocr)): continue
            features['paths'].append(os.path.join(src, image))

            # DOCUMENT REGIONS
            root = ET.parse(os.path.join(src, file_gt)).getroot()
            regions = []
            for parent in root:
                if parent.tag.split("}")[1] == 'Page':
                    for child in parent:
                        region_label = child[0].attrib['value']
                        region_bbox = [int(child[1].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                    int(child[1].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                        regions.append([region_label, region_bbox])

            # DOCUMENT TOKENS
            root = ET.parse(os.path.join(src, file_ocr)).getroot()
            tokens_bbox = []
            tokens_text = []
            nl = []
            center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)
            for parent in root:
                if parent.tag.split("}")[1] == 'Page':
                    for child in parent:
                        if child.tag.split("}")[1] == 'TextRegion':
                            for elem in child:
                                if elem.tag.split("}")[1] == 'TextLine':
                                    for word in elem:
                                        if word.tag.split("}")[1] == 'Word':
                                            word_bbox = [int(word[0].attrib['points'].split(" ")[0].split(",")[0].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[1].split(",")[1].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[2].split(",")[0].split(".")[0]),
                                                        int(word[0].attrib['points'].split(" ")[3].split(",")[1].split(".")[0])]
                                            word_text = word[1][0].text
                                            c = center(word_bbox)
                                            for reg in regions:
                                                r = reg[1]
                                                if r[0] < c[0] < r[2] and r[1] < c[1] < r[3]:
                                                    word_label = reg[0]
                                                    break
                                            tokens_bbox.append(word_bbox)
                                            tokens_text.append(word_text)
                                            nl.append(word_label)

            features['boxs'].append(tokens_bbox)
            features['texts'].append(tokens_text)
            node_labels.append(nl)

            # getting edges
            if self.edge_type == 'fully':
                u, v = self.fully_connected(range(len(tokens_bbox)))
            elif self.edge_type == 'knn':
                u,v = self.__knn(Image.open(os.path.join(src, image)).size, tokens_bbox)
            else:
                raise Exception('Other edge types still under development.')

            el = list()
            for e in zip(u, v):
                if (nl[e[0]] == nl[e[1]]) and (nl[e[0]] == 'positions' or nl[e[0]] == 'total'):
                    el.append('table')
                else: el.append('none')
            edge_labels.append(el)

            g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(tokens_bbox), idtype=torch.int32)
            graphs.append(g)

        return graphs, node_labels, edge_labels, features

    def __fromFUNSD(self, src : str) -> Tuple[list, list, list, list]:
        """Parsing FUNSD annotation files

        Args:
            src (str) : path to where data is stored

        Returns:
            tuple (lists) : graphs, nodes and edge labels, features
        """

        graphs, node_labels, edge_labels = list(), list(), list()
        features = {'paths': [], 'texts': [], 'boxs': []}
        # justOne = random.choice(os.listdir(os.path.join(src, 'adjusted_annotations'))).split(".")[0]

        if self.node_granularity == 'gt':
            for file in tqdm(os.listdir(os.path.join(src, 'adjusted_annotations')), desc='Creating graphs - GT'):

                img_name = f'{file.split(".")[0]}.png'
                img_path = os.path.join(src, 'images', img_name)
                features['paths'].append(img_path)

                with open(os.path.join(src, 'adjusted_annotations', file), 'r', encoding='utf-8') as f:
                    form = json.load(f)['form']

                # getting infos
                boxs, texts, ids, nl = list(), list(), list(), list()
                pair_labels = list()

                for elem in form:
                    boxs.append(elem['box'])
                    texts.append(elem['text'])
                    nl.append(elem['label'])
                    ids.append(elem['id'])
                    [pair_labels.append(pair) for pair in elem['linking']]

                for p, pair in enumerate(pair_labels):
                    pair_labels[p] = [ids.index(pair[0]), ids.index(pair[1])]

                node_labels.append(nl)
                features['texts'].append(texts)
                features['boxs'].append(boxs)

                # getting edges
                if self.edge_type == 'fully':
                    u, v = self.fully_connected(range(len(boxs)))
                elif self.edge_type == 'knn':
                    u,v = self.__knn(Image.open(img_path).size, boxs)
                elif self.edge_type == 'adaptive':
                    u, v = self.__adaptive(Image.open(img_path).size, boxs, texts)
                else:
                    raise Exception('GraphBuilder exception: Other edge types still under development.')

                el = list()
                for e in zip(u, v):
                    edge = [e[0], e[1]]
                    if edge in pair_labels: el.append('pair')
                    else: el.append('none')
                edge_labels.append(el)

                # creating graph
                g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(boxs), idtype=torch.int32)
                graphs.append(g)

            #! DEBUG PURPOSES TO VISUALIZE RANDOM GRAPH IMAGE FROM DATASET
            if False:
                if justOne == file.split(".")[0]:
                    print("\n\n### EXAMPLE ###")
                    print("Savin example:", img_name)

                    edge_unique_labels = np.unique(el)
                    g.edata['label'] = torch.tensor([np.where(target == edge_unique_labels)[0][0] for target in el])

                    g = self.balance_edges(g, 3, int(np.where('none' == edge_unique_labels)[0][0]))

                    img_removed = Image.open(img_path).convert('RGB')
                    draw_removed = ImageDraw.Draw(img_removed)

                    for b, box in enumerate(boxs):
                        if nl[b] == 'header':
                            color = 'yellow'
                        elif nl[b] == 'question':
                            color = 'blue'
                        elif nl[b] == 'answer':
                            color = 'green'
                        else:
                            color = 'gray'
                        draw_removed.rectangle(box, outline=color, width=3)

                    u, v = g.all_edges()
                    labels = g.edata['label'].tolist()
                    u, v = u.tolist(), v.tolist()

                    center = lambda rect: ((rect[2]+rect[0])/2, (rect[3]+rect[1])/2)

                    num_pair = 0
                    num_none = 0

                    for p, pair in enumerate(zip(u,v)):
                        sc = center(boxs[pair[0]])
                        ec = center(boxs[pair[1]])
                        if labels[p] == int(np.where('pair' == edge_unique_labels)[0][0]):
                            num_pair += 1
                            color = 'violet'
                            draw_removed.ellipse([(sc[0]-4,sc[1]-4), (sc[0]+4,sc[1]+4)], fill = 'green', outline='black')
                            draw_removed.ellipse([(ec[0]-4,ec[1]-4), (ec[0]+4,ec[1]+4)], fill = 'red', outline='black')
                        else:
                            num_none += 1
                            color='gray'
                        draw_removed.line((sc,ec), fill=color, width=3)

                    print("Balanced Links: None {} | Key-Value {}".format(num_none, num_pair))
                    img_removed.save(f'esempi/FUNSD/{img_name}_removed_graph.png')

        elif self.node_granularity == 'yolo':
            path_preds = os.path.join(src, 'yolo_bbox')
            path_images = os.path.join(src, 'images')
            path_gts = os.path.join(src, 'adjusted_annotations')
            all_paths, all_preds, all_links, all_labels, all_texts = load_predictions(path_preds, path_gts, path_images)
            for f, img_path in enumerate(tqdm(all_paths, desc='Creating graphs - YOLO')):

                features['paths'].append(img_path)
                features['boxs'].append(all_preds[f])
                features['texts'].append(all_texts[f])
                node_labels.append(all_labels[f])
                pair_labels = all_links[f]

                # getting edges
                if self.edge_type == 'fully':
                    u, v = self.fully_connected(range(len(features['boxs'][f])))
                elif self.edge_type == 'knn':
                    u,v = self.__knn(Image.open(img_path).size, features['boxs'][f])
                else:
                    raise Exception('GraphBuilder exception: Other edge types still under development.')

                el = list()
                for e in zip(u, v):
                    edge = [e[0], e[1]]
                    if edge in pair_labels: el.append('pair')
                    else: el.append('none')
                edge_labels.append(el)

                # creating graph
                g = dgl.graph((torch.tensor(u), torch.tensor(v)), num_nodes=len(features['boxs'][f]), idtype=torch.int32)
                graphs.append(g)
        else:
            #TODO develop OCR too
            raise Exception('GraphBuilder Exception: only \'gt\' or \'yolo\' available for FUNSD.')


        return graphs, node_labels, edge_labels, features
