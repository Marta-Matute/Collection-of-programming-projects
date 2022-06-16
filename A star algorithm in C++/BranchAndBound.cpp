#include "pch.h"
#include "Graph.h"
#include <queue>
#include <iostream>
#include <iomanip> 
#include <list>
#include "GraphApplicationDlg.h"


using namespace std;

struct tracksEdges {
	list<CEdge*> edges;
	double length = 0.0;
};

class BranchNode {
public:
	BranchNode* m_pFather;
	CEdge* m_pEdge;
	double m_Length;
	double m_MinLength;
	double m_Weight; // Peso para no ordenar que no es una cota inferior ni superior (no sirve para podar)
	unsigned m_CntRef;
	int index;
	list<int> indexos_visitats;
	vector<int> indexos;
	int n_indexos_visitats = 0;
	double cotaSuperior;
	double cotaInferior;
public:
	BranchNode() {}
	BranchNode(CEdge* pEdge, CVertex* pDestination)
		: m_pFather(NULL)
		, m_Length(pEdge->m_Length)
		, m_MinLength(m_Length + pDestination->m_Point.Distance(pEdge->m_pDestination->m_Point))
		, m_pEdge(pEdge)
		, m_Weight(m_Length + pDestination->m_Point.Distance2(pEdge->m_pDestination->m_Point))
		, m_CntRef(1)
	{
	}
	BranchNode(BranchNode* pFather, CEdge* pEdge, double minLength, double weight)
		: m_pFather(pFather)
		, m_pEdge(pEdge)
		, m_Length(pFather->m_Length + pEdge->m_Length)
		, m_MinLength(minLength)
		, m_Weight(weight)
		, m_CntRef(1)
	{
		++m_pFather->m_CntRef;
	}
	void Unlink()
	{
		if (--m_CntRef == 0) {
			if (m_pFather) m_pFather->Unlink();
			delete this;
		}
	}
};
list<CEdge*> TrackGreedy(CVertex* origin, CVertex* destiny);

// Initialize Tracks ============================================

void loadDijkstraTracks(CGraph& graph, list<CVertex*> visits, vector<tracksEdges*>& tracks) {
	int num_visita = 0;
	int index = 0;
	list<CVertex*> visites = visits;
	//visites.pop_back();
	for (CVertex* v1 : visites) {
		DijkstraQueue(graph, v1);
		v1->num_visita = num_visita;
		++num_visita;
		for (CVertex* v2 : visits) {
			tracks[index] = new tracksEdges;
			if (v1 != v2) {
				tracks[index]->edges = TrackGreedy(v1, v2);
				tracks[index]->length = v2->m_DijkstraDistance;
			}
			++index;
		}
	}
}

// Priority Queue Comparator ===================================

struct comparatorBN {
	bool operator()(const BranchNode* s1, const BranchNode* s2) {
		return s1->m_Weight > s2->m_Weight;
	}
};

// Initialize Node ============================================

void initializeNode(BranchNode* BB1actual, CVisits& visits, const int n) {
	BB1actual->m_Weight = 0;
	BB1actual->indexos_visitats.push_back(0);
	BB1actual->n_indexos_visitats += 1;
	BB1actual->index = 0;//visits.m_Vertices.front()->num_visita;
	BB1actual->indexos.resize(n, 0);
	BB1actual->indexos[BB1actual->index] = 1;
}

// Heuristic I =================================================
void Heuristica1(BranchNode*, const int, double, vector<tracksEdges*>&, list<int>&);


// Return result =================================================================

void returnTrack(CTrack& trackFinal, list<int>& millor_solucio, const int n, vector<tracksEdges*>& tracks) {
	for (list<int>::iterator it = millor_solucio.begin(); it != millor_solucio.end(); ++it) {
		trackFinal.m_Edges.splice(trackFinal.m_Edges.end(), tracks[*it]->edges);
	}
	for (int x = 0; x < n; ++x) delete tracks[x];
}

// SalesmanTrackBranchAndBound1 ===================================================

CTrack SalesmanTrackBranchAndBound1(CGraph& graph, CVisits& visits) {
	CTrack trackFinal(&graph);
	const int n = visits.GetNVertices();
	int dim = n * n;
	double cotaSuperior = DBL_MAX;

	list<int> millor_solucio;
	vector<tracksEdges*> tracks(dim);
	loadDijkstraTracks(graph, visits.m_Vertices, tracks);

	if (n > 2) {
		cotaSuperior = SalesmanTrackGreedy(graph, visits).Length() + 10e-5;
		BranchNode* BB1actual = new BranchNode;
		initializeNode(BB1actual, visits, n);
		Heuristica1(BB1actual, n, cotaSuperior, tracks, millor_solucio);
	}
	else for (int x = 0; x < n; ++x) { millor_solucio.push_back(x); }
	returnTrack(trackFinal, millor_solucio, n, tracks);
	return trackFinal;
}

void getMinsMatrix(vector<pair<double, double>>& lengths, const int n, vector<tracksEdges*>& tracks) {
	double cotamin = DBL_MAX, cotamax = 0;

	for (int col = 1; col < n; ++col) {
		for (int row = 0; row < n - 1; ++row) {
			if (row != col) {
				double lenactual = tracks[row * n + col]->length;
				if (lenactual < cotamin) cotamin = lenactual;
				if (lenactual > cotamax) cotamax = lenactual;
			}
		}
		lengths[col].first = cotamin;
		lengths[col].second = cotamax;
		cotamin = DBL_MAX; cotamax = 0;
	}
}


void dynamicMiniMax(vector<pair<double, double>>& lengths, const int n, BranchNode* BB3actual, vector<tracksEdges*>& tracks, int final, int i) {

	double cotaMin = DBL_MAX, cotaMax = 0;
	for (int col = 1; col < n; ++col) {
		if (BB3actual->indexos[col] || col == i) continue;
			cotaMin = DBL_MAX; cotaMax = 0;
			for (int row = 1; row < final; ++row) {
				if (row == col || (row == i && col == n - 1)) continue;
				double length = tracks[row * n + col]->length;
				if (length < cotaMin) cotaMin = length;
				if (length > cotaMax) cotaMax = length;
			}
		lengths[col].first = cotaMin;
		lengths[col].second = cotaMax;
	}
}


void recalculateDistances(/*BranchNode* BB3actual, */double& cotamin, double& cotamax, vector<pair<double, double>>& lengths, const int n) {
	cotamin = 0, cotamax = 0;
	for (int x = 1; x < n; ++x) {
		cotamin += lengths[x].first;
		cotamax += lengths[x].second;
	}
	cotamax += 1e-5;
}

// SalesmanTrackBranchAndBound2 ===================================================

void Heuristica2(BranchNode*, const int, double, vector<tracksEdges*>&, list<int>&, vector<pair<double, double>>&);

CTrack SalesmanTrackBranchAndBound2(CGraph& graph, CVisits& visits) {
	CTrack trackFinal(&graph);
	const int n = visits.GetNVertices();
	int dim = n * n;
	double cotaSuperior;

	list<int> millor_solucio;
	vector<tracksEdges*> tracks(dim);

	loadDijkstraTracks(graph, visits.m_Vertices, tracks);

	if (n > 2) {
		vector<pair<double, double>> lengths(n, make_pair(0, 0));
		getMinsMatrix(lengths, n, tracks);
		BranchNode* BB2actual = new BranchNode;
		initializeNode(BB2actual, visits, n);
		double cotamin = 0, cotamax = 0;
		for (int x = 1; x < n; ++x) {
			cotamin += lengths[x].first;
			cotamax += lengths[x].second;
		}
		BB2actual->m_Weight = 0;
		BB2actual->cotaInferior = cotamin;
		BB2actual->cotaSuperior = cotamax + 1e-5;
		cotaSuperior = BB2actual->cotaSuperior;
		Heuristica2(BB2actual, n, cotaSuperior, tracks, millor_solucio, lengths);
	}
	else { for (int x = 0; x < n; ++x) { millor_solucio.push_back(x); } }
	returnTrack(trackFinal, millor_solucio, n, tracks);
	return trackFinal;
}

// SalesmanTrackBranchAndBound3 ===================================================

void Heuristica3(BranchNode*, const int, double, vector<tracksEdges*>&, list<int>&, vector<pair<double, double>>&);

CTrack SalesmanTrackBranchAndBound3(CGraph& graph, CVisits& visits) {
	CTrack trackFinal(&graph);

	const int n = visits.GetNVertices();
	int dim = n * n;
	double cotaSuperior;

	list<int> millor_solucio;
	vector<tracksEdges*> tracks(dim);

	loadDijkstraTracks(graph, visits.m_Vertices, tracks);

	if (n > 2) {
		vector<pair<double, double>> lengths(n, make_pair(0, 0));
		double cotamin = 0, cotamax = 0;
		BranchNode* BB3actual = new BranchNode;
		initializeNode(BB3actual, visits, n);
		BB3actual->cotaInferior = 0;
		BB3actual->cotaSuperior = 0;
		dynamicMiniMax(lengths, n, BB3actual, tracks, n, NULL);
		recalculateDistances(cotamin, cotamax, lengths, n);
		cotaSuperior = cotamax;
		Heuristica3(BB3actual, n, cotaSuperior, tracks, millor_solucio, lengths);
	}
	else { for (int x = 0; x < n; ++x) { millor_solucio.push_back(x); } }
	returnTrack(trackFinal, millor_solucio, n, tracks);
	return trackFinal;
}



// ==================================================================================
// Heuristics =======================================================================
// ==================================================================================


// Heuristic I ======================================================================

void Heuristica1(BranchNode* BB1actual, const int n, double cotaSuperior, vector<tracksEdges*>& tracks, list<int>& millor_solucio) {
	priority_queue<BranchNode*, vector<BranchNode*>, comparatorBN> PQ;
	PQ.push(BB1actual);

	while (!PQ.empty()) {
		BB1actual = PQ.top(); PQ.pop();

		if (BB1actual->index != n - 1) {
			int vis;
			if (BB1actual->indexos_visitats.size() < n - 1) vis = n - 1;
			else vis = n;

			for (int i = 1; i < vis; ++i) {
				if (i != BB1actual->index && !BB1actual->indexos[i]) {
					int idx = BB1actual->index * n + i;
					double weight = BB1actual->m_Weight + tracks[idx]->length;
					if (weight < cotaSuperior) {
						BranchNode* newNode = new BranchNode;
						newNode->indexos_visitats = BB1actual->indexos_visitats;
						newNode->indexos_visitats.push_back(idx);

						newNode->indexos = BB1actual->indexos;
						newNode->indexos[i] = 1;

						newNode->index = i;
						newNode->m_Weight = weight;
						PQ.push(newNode);
					}
				}
			}
		}
		else if (BB1actual->indexos_visitats.size() == n) {
			millor_solucio = BB1actual->indexos_visitats;
			cotaSuperior = BB1actual->m_Weight;
			delete BB1actual;
			// while (!PQ.empty()) { BB1actual = PQ.top(); PQ.pop(); delete BB1actual; }
			return;
		}
		delete BB1actual;
	}
}


// Comparator Struct ================================================================
struct comparatorBNII {
	bool operator()(const BranchNode* s1, const BranchNode* s2) {
		return s1->cotaInferior > s2->cotaInferior;
	}
};


// Heuristic II =====================================================================

void Heuristica2(BranchNode* BB2actual, const int n, double cotaSuperior,
	vector<tracksEdges*>& tracks, list<int>& millor_solucio, vector<pair<double, double>>& lengths) {

	priority_queue<BranchNode*, vector<BranchNode*>, comparatorBNII> PQ;
	PQ.push(BB2actual);
	double best_weight = DBL_MAX;
	int vis;
	while (!PQ.empty()) {
		BB2actual = PQ.top(); PQ.pop();

		if (BB2actual->n_indexos_visitats == n) {
			millor_solucio = BB2actual->indexos_visitats;
			return;
		}

		if (BB2actual->n_indexos_visitats < n - 1) vis = n - 1;
		else vis = n;

		for (int i = 1; i < vis; ++i) {
			if (i != BB2actual->index && !BB2actual->indexos[i]) {
				int idx = BB2actual->index * n + i;

				double cotaMinima = BB2actual->cotaInferior - lengths[i].first + tracks[idx]->length;

				if (cotaMinima < cotaSuperior) {
					BranchNode* newNode = new BranchNode;
					newNode->indexos_visitats = BB2actual->indexos_visitats;
					newNode->indexos_visitats.push_back(idx);
					newNode->n_indexos_visitats = BB2actual->n_indexos_visitats + 1;
					newNode->m_Weight = BB2actual->m_Weight + tracks[idx]->length;
					newNode->index = i;
					newNode->indexos = BB2actual->indexos;
					newNode->indexos[i] = 1;

					newNode->cotaInferior = cotaMinima;
					newNode->cotaSuperior = BB2actual->cotaSuperior - lengths[i].second + tracks[idx]->length + 1e-5;

					PQ.push(newNode);
					double cotaMaxima = newNode->cotaSuperior;
					if (cotaMaxima < cotaSuperior) cotaSuperior = cotaMaxima;
				}
			}
		}
		delete BB2actual;
	}

}


// Heuristica III ===================================================================

void Heuristica3(BranchNode* BB3actual, const int n, double cotaSuperior, vector<tracksEdges*>& tracks, list<int>& millor_solucio, vector<pair<double, double>>& lengths) {
	priority_queue<BranchNode*, vector<BranchNode*>, comparatorBNII> PQ;
	PQ.push(BB3actual);
	double cotamin = 0, cotamax = 0;
	int vis;
	while (!PQ.empty()) {
		BB3actual = PQ.top(); PQ.pop();

		if (BB3actual->indexos_visitats.size() == n) {
			millor_solucio = BB3actual->indexos_visitats;
			cotaSuperior = BB3actual->m_Weight;
			delete BB3actual;
			return;
		}

		if (BB3actual->indexos_visitats.size() < n - 1) vis = n - 1;
		else vis = n;

		vector<pair<double, double>> lengths(n, make_pair(0, 0));

		

		for (int i = 1; i < vis; ++i) {
			if (i != BB3actual->index && !BB3actual->indexos[i]) {
				int idx = BB3actual->index * n + i;

				BranchNode* newNode = new BranchNode;
				dynamicMiniMax(lengths, n, BB3actual, tracks, n-1, i);
				recalculateDistances(cotamin, cotamax, lengths, n);
				
				newNode->indexos_visitats = BB3actual->indexos_visitats;
				newNode->indexos_visitats.push_back(idx);

				newNode->index = i;
				newNode->indexos = BB3actual->indexos;
				newNode->indexos[i] = 1;
				newNode->m_Weight = BB3actual->m_Weight + tracks[idx]->length;
				newNode->cotaInferior = cotamin + newNode->m_Weight;
				newNode->cotaSuperior = cotamax + newNode->m_Weight;
				//newNode->cotaSuperior = CS - BB3actual->cotaSuperior - lengths[i].second + tracks[idx]->length + 1e-5;

				/*double cotamin = 0, cotamax = 0;
				BB3actual->cotaSuperior = cotamax;
				BB3actual->cotaInferior = cotamin;
				cotaSuperior = cotamax;
				double CS = cotamax;*/

				//double cotaMinima = cotamin - BB3actual->cotaInferior - lengths[i].first + tracks[idx]->length;
				if (newNode->cotaSuperior < cotaSuperior) cotaSuperior = newNode->cotaSuperior;
				if (newNode->cotaInferior < cotaSuperior) {	
					PQ.push(newNode);
				}
			}
		}
		delete BB3actual;
	}
}

/*
double cotaSuperior = 0, cotaInferior = 0;
CTrack SalesmanTrackBranchAndBound3Model(CGraph& graph, CVisits& visits)
{
	CTrack trackResultant(&graph);
	int n = visits.GetNVertices();
	//int dim = n * n, index = 0;
	vector<vector<tracksEdges*>> tracks(n);
	for (int i = 0; i < n; ++i) {
		tracks[i].resize(n);
	}

	list<CVertex*> totalVisits = visits.m_Vertices;
	//totalVisits.pop_back();
	int num_visita = 0;
	int i = 0, j = 0, k = 0;
	for (CVertex* v1 : totalVisits) {
		DijkstraQueue(graph, v1);
		v1->num_visita = num_visita;
		++num_visita;
		for (CVertex* v2 : visits.m_Vertices) {
			tracks[i][j] = new tracksEdges;
			if (v1 != v2) {
				tracks[i][j]->edges = TrackGreedy(v1, v2);
				tracks[i][j]->length = v2->m_DijkstraDistance; //quiza no haga falta la ultima linea de la matriz, pero en su ejemplo esta
			}
			++j;
		}
		++i;
		j = 0;
	}
	double minim = DBL_MAX;
	double maxim = 0;
	if (n > 2) {
		cotaSuperior = 0;
		for (j = 1; j < n; ++j) {
			for (i = 0; i < n; ++i) {
				if (j != i) {
					if (minim > tracks[i][j]->length) minim = tracks[j][i]->length;
					if (maxim < tracks[i][j]->length) maxim = tracks[j][i]->length;
				}
			}
			cotaSuperior += maxim;
			cotaInferior += minim;
			minim = DBL_MAX;
			maxim = 0;
		}
	}

	double minimaCotaSuperior;
	double maximaCotaInferior;

	BranchNode* pActual = new BranchNode;
	pActual->index = visits.m_Vertices.front()->num_visita;
	pActual->indexos_visitats.push_back(0);
	pActual->m_Weight = 0;

	priority_queue<BranchNode*, vector<BranchNode*>, comparatorBB3> PQ;
	PQ.push(pActual);

	list<int> millor_solucio;
	BB1* millor_BBNode = new BB1;
	millor_BBNode->weight = DBL_MAX;
	list<int> llista_auxiliar;

	while (!PQ.empty()) {
		pActual = PQ.top();		PQ.pop();

		if (pActual->indexos_visitats.size() == n - 2 && pActual->weight < millor_BBNode->weight) {
			for (int x = 1; x < n - 1; ++x) {
				if (!inList(pActual->indexos_visitats, x)) {
					pActual->indexos_visitats.push_back(x);
					pActual->indexos_visitats.push_back(n - 1);
				}
			}
			millor_solucio = pActual->indexos_visitats;
			millor_BBNode = pActual;
		}
		for (i = 1; i < n - 1; ++i) {
			if (i != pActual->index && !inList(pActual->indexos_visitats, i)) {
				BB1* new_to_add = new BB1;
				for (k = 1; k < n; ++k) {
					if (inList(pActual->indexos_visitats, k) || k == i) continue;
					minim = DBL_MAX;
					maxim = 0;
					for (j = 1; j < n - 1; ++j) {
						if (j == k || (k == n - 1 && j == i) || inList(pActual->indexos_visitats, j)) continue; /// probablemente se puede mejorar mucho
						if (minim > tracks[j][k]->length) minim = tracks[j][k]->length;
						if (maxim < tracks[j][k]->length) maxim = tracks[j][k]->length;
					}
					new_to_add->cotaSuperior += maxim;
					new_to_add->cotaInferior += minim;
				}
				new_to_add->indexos_visitats = pActual->indexos_visitats;
				new_to_add->indexos_visitats.push_back(i);
				new_to_add->index = i;
				new_to_add->weight = pActual->weight + tracks[pActual->index][i]->length;
				new_to_add->cotaSuperior += new_to_add->weight;
				new_to_add->cotaInferior += new_to_add->weight;
				if (new_to_add->cotaSuperior < cotaSuperior) cotaSuperior = new_to_add->cotaSuperior;
				if (new_to_add->cotaInferior <= cotaSuperior) {
					PQ.push(new_to_add);
				}
			}
		}
	}

	int a, b;
	for (i = 0; i < n - 1; ++i) {
		a = millor_solucio.front();
		millor_solucio.pop_front();
		b = millor_solucio.front();
		trackResultant.m_Edges.splice(trackResultant.m_Edges.end(), tracks[b][a]->edges);
	}

	return trackResultant;
}*/