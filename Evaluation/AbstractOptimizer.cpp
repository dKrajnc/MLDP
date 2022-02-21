#include <Evaluation/AbstractOptimizer.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

AbstractOptimizer::AbstractOptimizer( QSettings* aSettings )
:
	mSettings( aSettings ),
	mModel( nullptr )
{
}

//-----------------------------------------------------------------------------

}
