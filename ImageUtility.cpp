#pragma once
#include "ImageUtility.h"
namespace ImageUtility {
	template <typename T>
	Image3D<T>* CreateMask(Image3D<T> * image) {
		Image3D<T>* mask = new Image3D<T>(*image, false);
		mask->setMinMax(BACKGROUND, EDGE);
		Thresholding<T>(image, mask, -1024, -400);
		ConnectedComponentLabeling<T>(mask);

		return mask;
	}


	template <typename T>
	Image3D<T>* Thresholding(Image3D<T> *image, Image3D<T> *mask, T minimum, T maximum) {
		#pragma omp parallel for
		for (int i = 0; i < image->getBufferSize(); i++) {
			auto value = image->getBuffer()[i];
			if (value >= minimum  && value <= maximum) {
				mask->getBuffer()[i] = FOREGROUND;
			}
			else {
				mask->getBuffer()[i] = BACKGROUND;
			}
		}

		qDebug() << "thresholding complete";
	}


	template <typename T>
	void ConnectedComponentLabeling(Image3D<T> * mask) {
		const short LIST_LEN = 32767;
		int *parent_list, *component_size;
		int neighbor_idx[3];
		int W, H, D, node_count, root_count;
		W = (int)(mask->getWidth());
		H = (int)(mask->getHeight());
		D = (int)(mask->getDepth());

		// change all xy borders to FOREGROUND
		#pragma omp parallel for
		for (int k = 0; k < D; k++) {
			for (int i = 0; i < W; i++) {
				mask->getBuffer()[mask->get3DIndex(i, 0, k)] = FOREGROUND;
				mask->getBuffer()[mask->get3DIndex(i, H - 1, k)] = FOREGROUND;
			}
			for (int j = 0; j < H; j++) {
				mask->getBuffer()[mask->get3DIndex(0, j, k)] = FOREGROUND;
				mask->getBuffer()[mask->get3DIndex(W - 1, j, k)] = FOREGROUND;
			}
		}

		// CCA first pass
		Image3D<T>* cca_label = new Image3D<T>(W + 1, H + 1, D + 1);
		cca_label->clear(BACKGROUND);
		parent_list = new int[LIST_LEN];
		component_size = new int[LIST_LEN];
		node_count = 0;
		for (int i = 1; i <= W; i++) {
			for (int j = 1; j <= H; j++) {
				for (int k = 1; k <= D; k++) {
					int idx = mask->get3DIndex(i - 1, j - 1, k - 1);
					auto value = mask->getBuffer()[idx];
					if (value == BACKGROUND) {
						continue;
					}

					neighbor_idx[0] = cca_label->get3DIndex(i, j - 1, k);
					neighbor_idx[1] = cca_label->get3DIndex(i - 1, j, k);
					neighbor_idx[2] = cca_label->get3DIndex(i, j, k - 1);
					int min_label = LIST_LEN;
					for (int n = 0; n < 3; n++) {
						auto neighbor_label = cca_label->getBuffer()[neighbor_idx[n]];
						if (neighbor_label >= 0 && min_label > neighbor_label) {
							min_label = neighbor_label;
						}
					}

					int cca_idx = cca_label->get3DIndex(i, j, k);
					if (min_label == LIST_LEN) {
						// add new component label
						parent_list[node_count] = node_count;
						cca_label->getBuffer()[cca_idx] = node_count;
						component_size[node_count] = 1;
						node_count++;
					}
					else {
						cca_label->getBuffer()[cca_idx] = min_label;
						component_size[min_label]++;

						for (int n = 0; n < 3; n++) {
							auto neighbor_label = cca_label->getBuffer()[neighbor_idx[n]];
							//union_components(parent_list, min_label, neighbor_label);
							if (neighbor_label >= 0) {
								if (parent_list[neighbor_label] > min_label) {
									parent_list[neighbor_label] = min_label;
								}
							}
						}
					}
				}
			}
		}

		// change parent list to root list, add component size to root
		#pragma omp parallel for
		for (int i = 0; i < node_count; i++) {
			if (i == parent_list[i])
				continue;

			int cur = i;
			while (parent_list[cur] != cur) {
				cur = parent_list[cur];
			}

			parent_list[i] = cur;
			component_size[cur] += component_size[i];
		}

		// CCA second pass
		#pragma omp parallel for
		for (int i = 1; i <= W; i++) {
			for (int j = 1; j <= H; j++) {
				for (int k = 1; k <= D; k++) {
					int cca_idx = cca_label->get3DIndex(i, j, k);
					auto label = cca_label->getBuffer()[cca_idx];
					if (label < 0)
						continue;

					int root = parent_list[label];
					cca_label->getBuffer()[cca_idx] = root;
				}
			}
		}

		// find max size component
		int max_size = 0;
		int max_idx;
		for (int i = 0; i < node_count; i++) {
			if (max_size < component_size[i] && parent_list[i] == i) {
				max_size = component_size[i];
				max_idx = i;
			}
		}

		// find second max size component
		component_size[max_idx] = 0;
		max_size = 0;
		for (int i = 0; i < node_count; i++) {
			if (max_size < component_size[i] && parent_list[i] == i) {
				max_size = component_size[i];
				max_idx = i;
			}
		}
		//qDebug() << "Voxel Count" << max_size;

		// CCA last pass
		#pragma omp parallel for
		for (int i = 1; i <= W; i++) {
			for (int j = 1; j <= H; j++) {
				for (int k = 1; k <= D; k++) {
					int cca_idx = cca_label->get3DIndex(i, j, k);
					auto label = cca_label->getBuffer()[cca_idx];
					if (label != max_idx) {
						int idx = mask->get3DIndex(i - 1, j - 1, k - 1);
						mask->getBuffer()[idx] = BACKGROUND;
					}
				}
			}
		}

		qDebug() << "CCL complete";
	}


	template <typename T>
	void FindEdge(Image3D<T> * image) {
		const int neighborX[] = { -1, 0, 1, 0 };
		const int neighborY[] = { 0, -1, 0, 1 };

		#pragma omp parallel for
		for (int z = 0; z < image->getDepth(); z++) {
			for (int y = 0; y < image->getHeight(); y++) {
				for (int x = 0; x < image->getWidth(); x++) {
					int index = image->get3DIndex(x, y, z);
					if (image->getBuffer()[index] != BACKGROUND) {
						for (int i = 0; i < 4; i++) {
							if (image->isValidIndex(x + neighborX[i], y + neighborY[i], z)) {
								int neighborIndex = image->get3DIndex(x + neighborX[i], y + neighborY[i], z);
								if (image->getBuffer()[neighborIndex] == BACKGROUND) {
									image->getBuffer()[index] = EDGE;
								}
							}
						}
					}
				}
			}
		}

		#pragma omp parallel for
		for (int z = 0; z < image->getDepth(); z++) {
			for (int y = 0; y < image->getHeight(); y++) {
				for (int x = 0; x < image->getWidth(); x++) {
					int index = image->get3DIndex(x, y, z);
					if (image->getBuffer()[index] == FOREGROUND) {
						image->getBuffer()[index] = BACKGROUND;
					}
				}
			}
		}

		qDebug() << "edge extraction complete";
	}


	template <typename T>
	Image3D<T>* CalculateChamferDistanceMap(Image3D<T>* image, int isothetic, int diagonal, int diagonal3D) {
		int W, H, D, a, b, c;
		W = (int)(image->getWidth());
		H = (int)(image->getHeight());
		D = (int)(image->getDepth());
		a = isothetic;
		b = diagonal;
		c = diagonal3D;
		const int neighborX[] = { -1,0,1, -1,0,1, -1,0,1, -1,0,1, -1,0 };
		const int neighborY[] = { -1,-1,-1,	0,0,0, 1,1,1, -1,-1,-1, 0,0 };
		const int neighborZ[] = { -1,-1,-1, -1,-1,-1, -1,-1,-1, 0,0,0, 0,0 };
		const int neighborD[] = { c,b,c, b,a,b, c,b,c, b,a,b, a,0 }; // approximate version of euclidean dist
		Image3D<T>* dist_map = new Image3D<T>(W + 2, H + 2, D + 2); // initialized to BACKGROUND

		// first pass
		for (int k = 1; k <= D; k++) {
			for (int j = 1; j <= H; j++) {
				for (int i = 1; i <= W; i++) {
					int dist_idx = dist_map->get3DIndex(i, j, k);
					int idx = image->get3DIndex(i - 1, j - 1, k - 1);
					auto value = image->getBuffer()[idx];

					if (value == EDGE) {
						dist_map->getBuffer()[dist_idx] = 0;
					}
					else {
						int min_dist = W + H + D; // any big value
						for (int n = 0; n < 14; n++) {
							int neighbor_idx = dist_map->get3DIndex(i + neighborX[n], j + neighborY[n], k + neighborZ[n]);
							auto neighbor_val = dist_map->getBuffer()[neighbor_idx];
							if (neighbor_val >= 0 && min_dist > neighbor_val + neighborD[n]) {
								min_dist = neighbor_val + neighborD[n];
							}
						}

						if (min_dist < W + H + D) {
							dist_map->getBuffer()[dist_idx] = min_dist;
						}
					}
				}
			}
		}

		// second pass
		Image3D<T>* dist_map_result = new Image3D<T>(W, H, D);
		for (int k = D; k > 0; k--) {
			for (int j = H; j > 0; j--) {
				for (int i = W; i > 0; i--) {
					int dist_idx = dist_map->get3DIndex(i, j, k);
					int idx = image->get3DIndex(i - 1, j - 1, k - 1);
					auto value = image->getBuffer()[idx];

					if (value != EDGE) {
						int min_dist = W + H + D; // any big value
						for (int n = 0; n < 14; n++) {
							int neighbor_idx = dist_map->get3DIndex(i - neighborX[n], j - neighborY[n], k - neighborZ[n]);
							auto neighbor_val = dist_map->getBuffer()[neighbor_idx];
							if (neighbor_val >= 0 && min_dist > neighbor_val + neighborD[n]) {
								min_dist = neighbor_val + neighborD[n];
							}
						}

						if (min_dist < W + H + D) {
							dist_map->getBuffer()[dist_idx] = min_dist;
						}
					}

					dist_map_result->getBuffer()[idx] = dist_map->getBuffer()[dist_idx];
				}
			}
		}

		qDebug() << "chamfer distance map complete";
		return dist_map_result;
	}


	template <typename T>
	int CalculateDistance(Image3D<T> * distanceMap, Image3D<T> * binaryMask) {
		std::vector<int> planeSum(binaryMask->getDepth(), 0);

		#pragma omp parallel for
		for (int z = 0; z < binaryMask->getDepth(); z++) {
			for (int y = 0; y < binaryMask->getHeight(); y++) {
				for (int x = 0; x < binaryMask->getWidth(); x++) {
					int index = binaryMask->get3DIndex(x, y, z);
					if (binaryMask->getBuffer()[index] == EDGE && distanceMap->isValidIndex(x, y, z)) {
						planeSum[z] += distanceMap->getBuffer()[distanceMap->get3DIndex(x, y, z)];
					}
				}
			}
		}

		int sum = 0;
		for (int partialSum : planeSum) {
			sum += partialSum;
		}

		return sum;
	}


	template <typename T>
	Image3D<T> * CalculateSubtractImage(Image3D<T> * lhs, Image3D<T> * rhs) {
		Image3D<T>* subtractedImage = new Image3D<T>(*lhs);

		int width = std::min(lhs->getWidth(), rhs->getWidth());
		int height = std::min(lhs->getHeight(), rhs->getHeight());
		int depth = std::min(lhs->getDepth(), rhs->getDepth());

		#pragma omp parallel for
		for (int z = 0; z < depth; z++) {
			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					subtractedImage->getBuffer()[subtractedImage->get3DIndex(x, y, z)] -= rhs->getBuffer()[rhs->get3DIndex(x, y, z)];
				}
			}
		}

		qDebug() << "subtraction complete";
		return subtractedImage;
	}
}