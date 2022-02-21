/*!
* \file
* PipelienTree class defitition. This file is part of Evaluation module.
* The PipelienTree class is responsible for generation of tree structure which contains all possible combinations of data preparation algorithms to for unique pipelines. 
* \remarks
*
* \authors
* dkrajnc
*/

#pragma once

#include <Evaluation/Export.h>
#include <QDebug>
#include <qsettings.h>
#include <fstream>
#include <random>

namespace dkeval
{

struct Node
{
	QVector< std::shared_ptr< Node > > children;
	std::shared_ptr< Node > parent;
	QString element;
	QList< QString > pool;
	int undersamplingCount;
	int oversamplingCount;
	int depth;
};

typedef QVector< std::shared_ptr< Node > > Creature;


class Evaluation_API PipelineTree
{

public:

	/*!
	* \brief Constructor to load settings parameters
	* \param [in] aSettings The settigns file
	*/
	PipelineTree( QSettings* aSettings )
	:
		mRoot( nullptr ),
		mTextFile(),
		mChoosenElements(),
		mElement(),
		mSettings( aSettings ),
		mIsInitValid( true ),
		mMaxAlgorithmRepetability(),
		mPool(),
		mMaxTreeDepth()
	{
		//Create parameters
		if ( mSettings == nullptr )
		{
			qDebug() << "PipelineTree - Error: Settings is a nullptr";
			mIsInitValid = false;
		}
		else
		{
			bool isMaxAlgorithmRepetability;
			mMaxAlgorithmRepetability = std::abs( mSettings->value( "Tree/maxAlgorithmRepetability" ).toInt( &isMaxAlgorithmRepetability ) );
			if ( !isMaxAlgorithmRepetability )
			{
				qDebug() << "PipelineTree - Error: Invalid parameter maxAlgorithmRepetability";
				mIsInitValid = false;
			}	

			bool isMaxTreeDepth;
			mMaxTreeDepth = std::abs( mSettings->value( "Tree/maxTreeDepth" ).toInt( &isMaxTreeDepth ) );
			if ( !isMaxTreeDepth )
			{
				qDebug() << "PipelineTree - Error: Invalid parameter maxTreeDepth";
				mIsInitValid = false;
			}
			
			bool isPool;
			mPool = mSettings->value( "Tree/pool" ).toStringList();
			if ( mPool.isEmpty() )
			{
				qDebug() << "PipelineTree - Error: Invalid parameter pool";
				mIsInitValid = false;
			}
		}
		
		std::random_device rd;
		mRng = new std::mt19937( rd() );
	}

	/*!
	* \brief Destructor
	*/
	~PipelineTree() {};

public:

	/*!
	* \brief Generates the tree structure of data preprocessing alogrithms
	*/
	void buildTree();

	/*!
	* \brief Equals operator.
	* \return std::shared_ptr< Node > of tree root node
	*/
	std::shared_ptr< Node > treeRoot(){ return mRoot; }

	/*!
	* \brief get node at index
	* \param [in] aPath Path of sequential algorithm pipeline in a tree
	* \param [in] aIndex Pipeline node index
	*/
	std::shared_ptr< Node > nodeAtIndex( Creature aPath, int aIndex );

	/*!
	* \brief get node at index
	* \param [in] aPath Path of sequential algorithm pipeline in a tree
	* \return QVector< QString > of algorithm names in the pipeline
	*/
	QVector< QString > algorithmNames( Creature aPath );		

	/*!
	* \brief get node at index
	* \param [in] aNode node in the algorithm pipeline
	* \return Creature List of children of algorithm pipeline node
	*/
	Creature children( std::shared_ptr< Node > aNode );	

	/*!
	* \brief generate random path to form the algorithm pipeline
	* \return Creature algorithm pipeline
	*/
	Creature randomPath();

	/*!
	* \brief Tests if the node is leaf
	* \param [in] aNode node in the algorithm pipeline
	* \return True if node is leaf, false otherwise.
	*/
	bool isLeaf( std::shared_ptr< Node > aNode );

	/*!
	* \brief Tests if the node is root
	* \param [in] aNode node in the algorithm pipeline
	* \return True if node is root, false otherwise.
	*/
	bool isRoot( std::shared_ptr< Node > aNode );	

	/*!
	* \brief Tests if the algorithm pipeline is valid
	* \param [in] aPath algorithm pipeline
	* \return True if pipeline is valid, false otherwise.
	*/
	bool isValidPath( Creature aPath );
	
	
private:
	
	PipelineTree();
	std::shared_ptr< Node > createNode( QString aAlgorithm, QList< QString >& aPool, int aUndersamplingCoung, int aOversmaplingCount, int aDepth, std::shared_ptr< Node > aParent );
	QList < int > indices( QList < QString >& aPool );
	void addNode( std::shared_ptr< Node > aNode );
	void applyConstrains( QList< QString >& aPool, QString aPreviousElement );
	void updateRepeatability( int& aUndersamplingCount, int& aOversamplingCount );
	void updatePool( QList < QString >& aPool, int& aUndersamplingCount, int& aOversamplingCount );	
	void printTree( std::shared_ptr< Node > aNode );
	void writeToFile();	
	Creature sibilings( std::shared_ptr< Node > aNode );
	std::shared_ptr< Node > parent( std::shared_ptr< Node > aNode );
	int randomIndex( int aListSize );
	QVector< int > algorithmIndices( QVector< std::shared_ptr< Node > >& aPath );

private:

	std::shared_ptr< Node > mRoot;
	std::ofstream mTextFile;
	QList< QString > mChoosenElements;
	QString mElement;
	std::mt19937* mRng;
	QSettings* mSettings;
	bool mIsInitValid;
	int mMaxAlgorithmRepetability;
	int mMaxTreeDepth;
	QStringList mPool;

};

}