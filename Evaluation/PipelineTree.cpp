#include <PipelineTree.h>

namespace dkeval
{
	//-----------------------------------------------------------------------------
	
	void PipelineTree::buildTree()
	{	
		//Initialize the root and build the tree 
		mPool << "addedLeaf"; //creates an empty node which determine the pipeline end if selected in a random path pipeline establishment.
							  //Handling of this artificial leaf node is done in applyConstrains and randomPath functions
		mRoot = createNode( "", mPool, 0, 0, 0, nullptr );		
		addNode( mRoot );
		
		//Remove artifical leaf from root children - handle in tree build
		auto rootChildrenSize = mRoot->children.size();
		mRoot->children.removeAt( rootChildrenSize - 1 );	
	}
	
	//-----------------------------------------------------------------------------
	
	std::shared_ptr< Node > PipelineTree::createNode( QString aAlgorithm, QList< QString >& aPool, int aUndersamplingCoung, int aOversmaplingCount, int aDepth, std::shared_ptr< Node > aParent )
	{
		std::shared_ptr< Node > node = std::make_shared< Node >();
		node->element                = aAlgorithm;
		node->pool                   = aPool;
		node->undersamplingCount     = aUndersamplingCoung;
		node->oversamplingCount      = aOversmaplingCount;
		node->parent                 = aParent;
		node->depth                  = aDepth;
			
		if ( aDepth < mMaxTreeDepth )
		{
			for ( int i = 0; i < aPool.size(); ++i )
			{
				node->children.push_back( nullptr );
			}
		}
			
	
		return node;
	}
	
	//-----------------------------------------------------------------------------
	
	void PipelineTree::addNode( std::shared_ptr< Node > aNode )
	{
		if ( aNode == nullptr )
		{
			qDebug() << "Error - the root is empty!";
		}
		else
		{		
			for ( int i = 0; i < aNode->children.size(); ++i )
			{
				if ( aNode->depth < mMaxTreeDepth )
				{
					if ( aNode->children.at( i ) == nullptr )
					{
						auto pool            = aNode->pool;
						auto previousElement = aNode->element;
						auto parent          = aNode->parent;
						auto nodeDepth       = aNode->depth;

						++nodeDepth;

						//Next node element
						mElement = pool.at( i );

						//handle Feature Selection and Dimensionality reduction duality, constrain 2 consecutive oversampling elements 
						applyConstrains( pool, previousElement );

						//Get undersampling and oversampling repetition number and update
						auto undersamplingCount = aNode->undersamplingCount;
						auto oversamplingCount  = aNode->oversamplingCount;

						//check and increment the count of undersampling and oversmapling		
						updateRepeatability( undersamplingCount, oversamplingCount );

						//Remove the elements which should not be saved in the next node from the pool
						updatePool( pool, undersamplingCount, oversamplingCount );

						//Create node and attach it to the tree					
						aNode->children[ i ] = createNode( mElement, pool, undersamplingCount, oversamplingCount, nodeDepth, aNode );
						addNode( aNode->children[ i ] );

					}
				}				
			}		
		}		
	}
	
	//-----------------------------------------------------------------------------
	
	void PipelineTree::applyConstrains( QList< QString >& aPool, QString aPreviousElement )
	{	
		if ( mElement == "FeatureSelection" )
		{
			auto index = aPool.indexOf( "PCA" );
			aPool.removeAt( index );		
		}
		else if ( mElement == "PCA" )
		{
			auto index = aPool.indexOf( "FeatureSelection" );
			aPool.removeAt( index );	
		}
		else if ( mElement == "Oversampling" && mElement == aPreviousElement ) //if present and previous elements are both oversampling, remove from the pool
		{                                                                      
			auto index = aPool.indexOf( mElement );
			aPool.removeAt( index );

			
			//Change the selection of current oversmapling element to other indexed element to avoid consecutive selection - IMPORTANT: only if 2 ot less samples are in the pool
			auto indices = this->indices( aPool ); //Get all indices of the pool
			
			if ( aPool.size() <= 2 ) 
			{
				for ( auto& id : indices ) //Change element based on index comparison
				{
					if ( id != index )
					{
						mElement = aPool.at( id );
					}
				}
			}					
		}	
		else if ( mElement == "addedLeaf" ) //removes all elements from the pool, hence it becomes an artificial leaf node
		{
			aPool.clear();
		}
	}
		
	//-----------------------------------------------------------------------------

	void PipelineTree::updateRepeatability( int& aUndersamplingCount, int& aOversamplingCount )
	{
		if ( mElement == "Undersampling" )
		{
			if ( aUndersamplingCount < mMaxAlgorithmRepetability )
			{
				aUndersamplingCount++;
			}			
		}
		else if ( mElement == "Oversampling" )
		{
			if ( aOversamplingCount < mMaxAlgorithmRepetability )
			{
				aOversamplingCount++;
			}
		}
	}

	//-----------------------------------------------------------------------------

	void PipelineTree::updatePool( QList < QString >& aPool, int& aUndersamplingCount, int& aOversamplingCount )
	{
		if ( mElement == "Undersampling" )
		{
			if ( aUndersamplingCount >= mMaxAlgorithmRepetability )
			{
				auto index = aPool.indexOf( mElement );
				aPool.removeAt( index );
			}
		}
		else if ( mElement == "Oversampling" )
		{
			if ( aOversamplingCount >= mMaxAlgorithmRepetability )
			{
				auto index = aPool.indexOf( mElement );
				aPool.removeAt( index );
			}
			else if ( aPool.size() == 1 && mElement == aPool.at( 0 ) )
			{
				auto index = aPool.indexOf( mElement );
				aPool.removeAt( index );
			}
		}
		else
		{
			auto index = aPool.indexOf( mElement );
			aPool.removeAt( index );
		}
	}

	//-----------------------------------------------------------------------------

	QList < int > PipelineTree::indices( QList < QString >& aPool )
	{
		QList< int > indices;

		for ( auto& element : aPool ) //Find all indices in a pool
		{
			auto index = aPool.indexOf( element );
			indices << index;
		}

		return indices;
	}

	//-----------------------------------------------------------------------------

	void PipelineTree::printTree( std::shared_ptr< Node > aNode )
	{
		if ( aNode == nullptr )
		{
			qDebug() << "Error - the root is empty!";
		}
		else
		{
			for ( int i = 0; i < aNode->children.size(); ++i )
			{				
					if ( aNode->element.isEmpty() == false )
					{
						mChoosenElements << aNode->element;
					}					

					if ( aNode->children.at( i )->children.isEmpty() == true ) //Check if it is a leaf
					{
						mChoosenElements << aNode->children.at( i )->element; //add leaf element into list

						writeToFile();

						mChoosenElements.clear();
					}

					printTree( aNode->children.at( i ) );				
			}
		}
	}

	//-----------------------------------------------------------------------------

	void PipelineTree::writeToFile()
	{
		for ( auto& element : mChoosenElements )
		{
			mTextFile << "-" << element.toStdString();
		}

		mTextFile << "\n";
	}

	//-----------------------------------------------------------------------------

	QVector< QString > PipelineTree::algorithmNames( Creature aPath )
	{		
		QVector< QString > names;

		for ( int i = 0; i < aPath.size(); ++i )
		{
			auto name = aPath.at( i )->element;
			
			names.push_back( name );			
		}

		return names;
	}

	//-----------------------------------------------------------------------------

	std::shared_ptr< Node > PipelineTree::nodeAtIndex( Creature aPath, int aIndex )
	{
		auto node = aPath.at( aIndex );

		return node;
	}

	//-----------------------------------------------------------------------------

	QVector< int > PipelineTree::algorithmIndices( QVector< std::shared_ptr< Node > >& aPath )
	{
		QVector< int > indices;

		for ( int i = 0; i < aPath.size(); ++i )
		{
			auto node = aPath.at( i );
			auto siblings = this->sibilings( node );
			auto index = siblings.indexOf( node );
			indices.push_back( index );
		}

		return indices;
	}

	//-----------------------------------------------------------------------------

	std::shared_ptr< Node > PipelineTree::parent( std::shared_ptr< Node > aNode )
	{
		if ( aNode != nullptr )
		{
			return aNode->parent;
		}
		else
		{			
			return nullptr;
		}
	}

	//-----------------------------------------------------------------------------

	Creature PipelineTree::children( std::shared_ptr< Node > aNode )
	{
		QVector < std::shared_ptr< Node > > children;

		if ( aNode == nullptr || isLeaf( aNode ) )
		{
			return children;
		}

		for ( int i = 0; i < aNode->children.size(); ++i )
		{
			children << aNode->children.at( i );
		}

		return children;
	}

	//-----------------------------------------------------------------------------

	Creature PipelineTree::sibilings( std::shared_ptr< Node > aNode )
	{
		QVector < std::shared_ptr< Node > > sibilings;

		if ( aNode == nullptr || isRoot( aNode ) )
		{
			return sibilings;
		}		

		for ( int i = 0; i < aNode->parent->children.count(); ++i )
		{
			sibilings << aNode->parent->children.at( i );
		}

		return sibilings;
	}

	//-----------------------------------------------------------------------------

	bool PipelineTree::isLeaf( std::shared_ptr< Node > aNode )
	{		
		if ( aNode == nullptr )
		{
			qDebug() << "PipelineTree::isLeaf received a nullptr argument";

			std::exit( EXIT_SUCCESS );
		}
		else if ( aNode->children.isEmpty() )
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	//-----------------------------------------------------------------------------

	bool PipelineTree::isRoot( std::shared_ptr< Node > aNode )
	{
		if ( aNode == nullptr )
		{
			qDebug() << "PipelineTree::isRoot received a nullptr argument";

			std::exit( EXIT_SUCCESS );
		}
		else if ( aNode->parent == nullptr )
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	//-----------------------------------------------------------------------------

	Creature PipelineTree::randomPath()
	{		
		QVector< std::shared_ptr< Node > > pathNodes;

		auto node = mRoot;		

		while ( true )
		{
			auto childrenOfNode = node->children;
			auto randomIndex    = this->randomIndex( childrenOfNode.size() );

			node = childrenOfNode.at( randomIndex );

			if ( node->element != "addedLeaf" ) //if aritificial leaf node is selected, don't include it in the pipeline
			{
				pathNodes.push_back( node );
			}

			if ( isLeaf( node ) ) break;
		}

		return pathNodes;
	}

	//-----------------------------------------------------------------------------

	bool PipelineTree::isValidPath( Creature aPath )
	{
		if ( aPath.isEmpty() )
		{
			qDebug() << "path is empty!";
		}

		auto parent = this->parent( aPath.at( 0 ) );

		for ( int i = 0; i < aPath.size(); ++i )
		{
			auto node = aPath.at( i );
			
			if ( !parent->children.contains( node ) )
			{
				qDebug() << "Error - node is not part of siblings!";

				return false;
				break;
			}

			parent = node;
		}

		return true;
	}

	//-----------------------------------------------------------------------------

	int PipelineTree::randomIndex( int aListSize )
	{
		std::uniform_int_distribution< int > dice( 0, aListSize - 1 );

		return dice( *mRng );
	}

	//-----------------------------------------------------------------------------

}