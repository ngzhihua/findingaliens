#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

/***********************************************************
  Helper functions 
 ***********************************************************/

//For exiting on error condition
void die(int lineNo);

//For trackinng execution
long long wallClockTime();


/***********************************************************
  Square matrix related functions, used by both world and pattern
 ***********************************************************/
char** allocateEmptySquareMatrix(int size);

char** allocateSquareMatrix( int size, char defaultValue );

void freeSquareMatrix( char** );

void printSquareMatrix( char**, int size );


/***********************************************************
  World  related functions
 ***********************************************************/

#define ALIVE 'X' 
#define DEAD 'O'

char** readWorldFromFile( char* fname, int* size );

int countNeighbours(char** world, int row, int col);

void evolveWorld(char** curWorld, char** nextWorld, int size);


/***********************************************************
  Simple circular linked list for match records
 ***********************************************************/

typedef struct MSTRUCT {
	int iteration, row, col, rotation;
	struct MSTRUCT *next;
} MATCH;


typedef struct {
	int nItem;
	MATCH* tail;
} MATCHLIST;

MATCHLIST* newList();

void deleteList( MATCHLIST*);

void insertEnd(MATCHLIST*, int, int, int, int);

void printList(MATCHLIST*);

//For conversion of match entries into array
void convertMatchToArray(MATCH* match, int* array, int i);

/***********************************************************
  Search related functions
 ***********************************************************/

//Using the compass direction to indicate the rotation of pattern
#define N 0 //no rotation
#define E 1 //90 degree clockwise
#define S 2 //180 degree clockwise
#define W 3 //90 degree anti-clockwise

char** readPatternFromFile( char* fname, int* size );

void rotate90(char** current, char** rotated, int size);

void searchPatterns(char** world, int wSize, int iteration, 
		char** patterns[4], int pSize, MATCHLIST* list);

void searchSinglePattern(char** world, int wSize, int interation,
		char** pattern, int pSize, int rotation, MATCHLIST* list);

/***********************************************************
  Main function
 ***********************************************************/


int main( int argc, char** argv)
{
	char **curW, **nextW, **temp, dummy[20];
	char **patterns[4];
	int dir, iterations, iter;
	int size, patternSize;
	long long before, after;
	//Added variables
	int numtasks, rank, dest = 1, source, rc, count, tag=1, rowIndex, sendSize, recvSize = 0;
	char **bufferedWorld;
	MATCHLIST* bufferedList = newList();
	int * matchResult;
	MPI_Status Stat;
	MATCHLIST*list;

	if (argc < 4 ){
		fprintf(stderr, 
				"Usage: %s <world file> <Iterations> <pattern file>\n", argv[0]);
		exit(1);
	} 


	curW = readWorldFromFile(argv[1], &size);
	nextW = allocateSquareMatrix(size+2, DEAD);


	printf("World Size = %d\n", size);

	iterations = atoi(argv[2]);
	printf("Iterations = %d\n", iterations);

	patterns[N] = readPatternFromFile(argv[3], &patternSize);
	for (dir = E; dir <= W; dir++){
		patterns[dir] = allocateSquareMatrix(patternSize, DEAD);
		rotate90(patterns[dir-1], patterns[dir], patternSize);
	}
	printf("Pattern size = %d\n", patternSize);

#ifdef DEBUG
	printSquareMatrix(patterns[N], patternSize);
	printSquareMatrix(patterns[E], patternSize);
	printSquareMatrix(patterns[S], patternSize);
	printSquareMatrix(patterns[W], patternSize);
#endif


	//Start timer
	before = wallClockTime();

	//Actual work start
	list = newList();

	//Start paralellizing
	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);



	//if (rank == 0){
	//	int i;
	//	for (i = 1; i < 8; i++){
	//		MPI_Send(patterns, patternSize * patternSize * 4, MPI_CHAR, i, 0, MPI_COMM_WORLD);
	//			printf("send pattern successful\n");
	//		}	
	//}
	//else{
	//	MPI_Recv(patterns, patternSize * patternSize * 4, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &Stat);	
	//	printf("recv pattern successful\n");
	//	printSquareMatrix(patterns[N], patternSize);
	//	printSquareMatrix(patterns[E], patternSize);
	//	printSquareMatrix(patterns[S], patternSize);
	//	printSquareMatrix(patterns[W], patternSize);
	//}

	for (iter = 0; iter < iterations; iter++){

#ifdef DEBUG
		printf("World Iteration.%d\n", iter);
		printSquareMatrix(curW, size+2);
#endif
		if (rank == 0){
			if (dest % numtasks == 0 ){
				int i;
				for (i = 1; i < (numtasks); i++){
					MPI_Probe(i, tag, MPI_COMM_WORLD, &Stat);
					MPI_Get_count(&Stat, MPI_INT, &recvSize);
					int* recvBuffer = (int *)malloc(sizeof(int) * recvSize);
					MPI_Recv(recvBuffer, recvSize, MPI_INT, i, tag, MPI_COMM_WORLD, &Stat);
					int j;    
					for ( j = 0; j < recvSize; j += 4){
						insertEnd(list, recvBuffer[j], recvBuffer[j + 1], recvBuffer[j + 2], recvBuffer[j + 3]);
						printf("%d %d %d %d\n", recvBuffer[j], recvBuffer[j+1], recvBuffer[j+2], recvBuffer[j+3]);
					}
					free(recvBuffer);
				}
				tag++;
				dest++;
			}
			rc = MPI_Send(&(curW[0][0]), (size + 2) * (size + 2), MPI_CHAR, dest % 8, tag, MPI_COMM_WORLD);
			printf("send successful to %d\n", dest);
			evolveWorld(curW, nextW, size);
			temp = curW;
			curW = nextW;
			nextW = temp;
			dest ++;
		}
		else if((rank-1) == (iter%(numtasks -1))){
			bufferedWorld = allocateEmptySquareMatrix(size + 2);
			//if (rank == 1)
			//printSquareMatrix(bufferedWorld, size + 2);
			//printf("local tag is %d\n", tag);
			rc = MPI_Recv(&(bufferedWorld[0][0]), (size + 2) * (size + 2), MPI_CHAR, 0, tag, MPI_COMM_WORLD, &Stat);
			bufferedList = newList();
			//printf("rank %d recv successful %d characters of data and tag is %d\n", rank, Stat.count,Stat.MPI_TAG);
			//printf("ERROR: %d\n", Stat.MPI_ERROR);
			//printf("rank %d pattern size is %d\n", rank, patternSize);
			//printf("searching pattern for rank %d\n", rank);
			searchPatterns(bufferedWorld, size, iter, patterns, patternSize, bufferedList);
			//printf("Number of items in bufferedList : %d\n", bufferedList->nItem);
			matchResult = (int *) malloc(bufferedList->nItem * 4 * sizeof(int));
			//printf("matched results\n");
			//printf("rank %d converting match list to array\n", rank);
			printList(bufferedList);
			MATCH* curr = bufferedList->tail->next;
			int i;
			for ( i = 0; i < bufferedList->nItem; curr = curr->next){
				convertMatchToArray(curr, matchResult, i);
				i++;
			}
			printf("match result item:  %d\n", bufferedList->nItem *4, tag);
			for (i = 0; i < bufferedList->nItem * 4; i++){
				printf("%d ", matchResult[i]);
			}
			rc = MPI_Send(&(matchResult[0]), bufferedList->nItem * 4, MPI_INT, 0, tag, MPI_COMM_WORLD);
			deleteList(bufferedList);
			tag++;
		}
		//MPI_Barrier(MPI_COMM_WORLD);
		// searchPatterns( curW, size, iter, patterns, patternSize, list);

		// //Generate next generation
		// evolveWorld( curW, nextW, size );
		// temp = curW;
		// curW = nextW;
		// nextW = temp;
	}
	if (rank == 0){
		printf("rank 0's dest is %d",dest);
		int i;
		for (i = 1; i < (dest) % (numtasks); i++){
			MPI_Probe(i, tag, MPI_COMM_WORLD, &Stat);
			MPI_Get_count(&Stat, MPI_INT, &recvSize);
			int* recvBuffer = (int *)malloc(sizeof(int) * recvSize);
			MPI_Recv(recvBuffer, recvSize, MPI_INT, i, tag, MPI_COMM_WORLD, &Stat);
			int j;    
			for ( j = 0; j < recvSize; j += 4){
				insertEnd(list, recvBuffer[j], recvBuffer[j + 1], recvBuffer[j + 2], recvBuffer[j + 3]);
				printf("%d %d %d %d\n", recvBuffer[j], recvBuffer[j+1], recvBuffer[j+2], recvBuffer[j+3]);
			}
			free(recvBuffer);
		}


	}	
	MPI_Barrier(MPI_COMM_WORLD);
	if (rank == 0){
		printList(list);
		MPI_Finalize();	
		after = wallClockTime();

		printf("Sequential SETL took %1.2f seconds\n", 
				((float)(after - before))/1000000000);

	}

	//printList( list );

	//Stop timer

	//Clean up
	deleteList( list );

	freeSquareMatrix( curW );
	freeSquareMatrix( nextW );

	freeSquareMatrix( patterns[0] );
	freeSquareMatrix( patterns[1] );
	freeSquareMatrix( patterns[2] );
	freeSquareMatrix( patterns[3] );

	return 0;
}

/***********************************************************
  Helper functions 
 ***********************************************************/
void convertMatchToArray(MATCH* match, int* array, int i){
	array[i*4+1] = match->iteration;
	array[i*4+2] = match->row;
	array[i*4+3] = match->col;
	array[i*4+4] = match->rotation;
}

void die(int lineNo)
{
	fprintf(stderr, "Error at line %d. Exiting\n", lineNo);
	exit(1);
}

long long wallClockTime( )
{
#ifdef __linux__
	struct timespec tp;
	clock_gettime(CLOCK_REALTIME, &tp);
	return (long long)(tp.tv_nsec + (long long)tp.tv_sec * 1000000000ll);
#else
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_usec * 1000 + (long long)tv.tv_sec * 1000000000ll);
#endif
}

/***********************************************************
  Square matrix related functions, used by both world and pattern
 ***********************************************************/
char ** allocateEmptySquareMatrix(int size){

	char* contiguous;
	char** matrix;
	int i;

	//Using a least compiler version dependent approach here
	//C99, C11 have a nicer syntax.    
	contiguous = (char*) malloc(sizeof(char) * size * size);
	if (contiguous == NULL) 
		die(__LINE__);

	//Point the row array to the right place
	matrix = (char**) malloc(sizeof(char*) * size );
	if (matrix == NULL) 
		die(__LINE__);

	matrix[0] = contiguous;
	for (i = 1; i < size; i++){
		matrix[i] = &contiguous[i*size];
	}

	return matrix;
}


char** allocateSquareMatrix( int size, char defaultValue )
{

	char* contiguous;
	char** matrix;
	int i;

	//Using a least compiler version dependent approach here
	//C99, C11 have a nicer syntax.    
	contiguous = (char*) malloc(sizeof(char) * size * size);
	if (contiguous == NULL) 
		die(__LINE__);


	memset(contiguous, defaultValue, size * size );

	//Point the row array to the right place
	matrix = (char**) malloc(sizeof(char*) * size );
	if (matrix == NULL) 
		die(__LINE__);

	matrix[0] = contiguous;
	for (i = 1; i < size; i++){
		matrix[i] = &contiguous[i*size];
	}

	return matrix;
}

void printSquareMatrix( char** matrix, int size )
{
	int i,j;

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			printf("%c", matrix[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void freeSquareMatrix( char** matrix )
{
	if (matrix == NULL) return;

	free( matrix[0] );
}

/***********************************************************
  World  related functions
 ***********************************************************/

char** readWorldFromFile( char* fname, int* sizePtr )
{
	FILE* inf;

	char temp, **world;
	int i, j;
	int size;

	inf = fopen(fname,"r");
	if (inf == NULL)
		die(__LINE__);


	fscanf(inf, "%d", &size);
	fscanf(inf, "%c", &temp);

	//Using the "halo" approach
	// allocated additional top + bottom rows
	// and leftmost and rightmost rows to form a boundary
	// to simplify computation of cell along edges
	world = allocateSquareMatrix( size + 2, DEAD );

	for (i = 1; i <= size; i++){
		for (j = 1; j <= size; j++){
			fscanf(inf, "%c", &world[i][j]);
		}
		fscanf(inf, "%c", &temp);
	}

	*sizePtr = size;    //return size
	return world;

}

int countNeighbours(char** world, int row, int col)
	//Assume 1 <= row, col <= size, no check 
{
	int i, j, count;

	count = 0;
	for(i = row-1; i <= row+1; i++){
		for(j = col-1; j <= col+1; j++){
			count += (world[i][j] == ALIVE );
		}
	}

	//discount the center
	count -= (world[row][col] == ALIVE);

	return count;

}

void evolveWorld(char** curWorld, char** nextWorld, int size)
{
	int i, j, liveNeighbours;

	for (i = 1; i <= size; i++){
		for (j = 1; j <= size; j++){
			liveNeighbours = countNeighbours(curWorld, i, j);
			nextWorld[i][j] = DEAD;

			//Only take care of alive cases
			if (curWorld[i][j] == ALIVE) {

				if (liveNeighbours == 2 || liveNeighbours == 3)
					nextWorld[i][j] = ALIVE;

			} else if (liveNeighbours == 3)
				nextWorld[i][j] = ALIVE;
		} 
	}
}

/***********************************************************
  Search related functions
 ***********************************************************/

char** readPatternFromFile( char* fname, int* sizePtr )
{
	FILE* inf;

	char temp, **pattern;
	int i, j;
	int size;

	inf = fopen(fname,"r");
	if (inf == NULL)
		die(__LINE__);


	fscanf(inf, "%d", &size);
	fscanf(inf, "%c", &temp);

	pattern = allocateSquareMatrix( size, DEAD );

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			fscanf(inf, "%c", &pattern[i][j]);
		}
		fscanf(inf, "%c", &temp);
	}

	*sizePtr = size;    //return size
	return pattern;
}


void rotate90(char** current, char** rotated, int size)
{
	int i, j;

	for (i = 0; i < size; i++){
		for (j = 0; j < size; j++){
			rotated[j][size-i-1] = current[i][j];
		}
	}
}

void searchPatterns(char** world, int wSize, int iteration, 
		char** patterns[4], int pSize, MATCHLIST* list)
{
	int dir;

	for (dir = N; dir <= W; dir++){
		searchSinglePattern(world, wSize, iteration, 
				patterns[dir], pSize, dir, list);
	}

}

void searchSinglePattern(char** world, int wSize, int iteration,
		char** pattern, int pSize, int rotation, MATCHLIST* list)
{
	int wRow, wCol, pRow, pCol, match;	


	for (wRow = 1; wRow <= (wSize-pSize+1); wRow++){
		for (wCol = 1; wCol <= (wSize-pSize+1); wCol++){
			match = 1;
#ifdef DEBUGMORE
			printf("S:(%d, %d)\n", wRow-1, wCol-1);
#endif
			for (pRow = 0; match && pRow < pSize; pRow++){
				for (pCol = 0; match && pCol < pSize; pCol++){
					if(world[wRow+pRow][wCol+pCol] != pattern[pRow][pCol]){
#ifdef DEBUGMORE
						printf("\tF:(%d, %d) %c != %c\n", pRow, pCol,
								world[wRow+pRow][wCol+pCol], pattern[pRow][pCol]);
#endif
						match = 0;    
					}
				}
			}
			if (match){
				insertEnd(list, iteration, wRow-1, wCol-1, rotation);
#ifdef DEBUGMORE
				printf("*** Row = %d, Col = %d\n", wRow-1, wCol-1);
#endif
			}
		}
	}
}

/***********************************************************
  Simple circular linked list for match records
 ***********************************************************/

MATCHLIST* newList()
{
	MATCHLIST* list;

	list = (MATCHLIST*) malloc(sizeof(MATCHLIST));
	if (list == NULL)
		die(__LINE__);

	list->nItem = 0;
	list->tail = NULL;

	return list;
}

void deleteList( MATCHLIST* list)
{
	MATCH *cur, *next;
	int i;
	//delete items first

	if (list->nItem != 0 ){
		cur = list->tail->next;
		next = cur->next;
		for( i = 0; i < list->nItem; i++, cur = next, next = next->next ) {
			free(cur); 
		}

	}
	free( list );
}

void insertEnd(MATCHLIST* list, 
		int iteration, int row, int col, int rotation)
{
	MATCH* newItem;

	newItem = (MATCH*) malloc(sizeof(MATCH));
	if (newItem == NULL)
		die(__LINE__);

	newItem->iteration = iteration;
	newItem->row = row;
	newItem->col = col;
	newItem->rotation = rotation;

	if (list->nItem == 0){
		newItem->next = newItem;
		list->tail = newItem;
	} else {
		newItem->next = list->tail->next;
		list->tail->next = newItem;
		list->tail = newItem;
	}

	(list->nItem)++;

}

void printList(MATCHLIST* list)
{
	int i;
	MATCH* cur;

	printf("List size = %d\n", list->nItem);    


	if (list->nItem == 0) return;

	cur = list->tail->next;
	for( i = 0; i < list->nItem; i++, cur=cur->next){
		printf("%d:%d:%d:%d\n", 
				cur->iteration, cur->row, cur->col, cur->rotation);
	}
}

