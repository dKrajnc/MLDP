#include <Evaluation/AbstractModel.h>

namespace lpmleval
{

//-----------------------------------------------------------------------------

AbstractModel::AbstractModel( QSettings* aSettings )
:
	mSettings( aSettings ),
	mFeatureNames()
{
}

//-----------------------------------------------------------------------------

AbstractModel::~AbstractModel()
{
}

//-----------------------------------------------------------------------------

}
