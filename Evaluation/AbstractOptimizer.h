#pragma once

#include <Evaluation/Export.h>
#include <Evaluation/AbstractModel.h>
#include <QSettings>
#include <QString>

namespace lpmleval
{

//-----------------------------------------------------------------------------

class Evaluation_API AbstractOptimizer
{

public:

	AbstractOptimizer( QSettings* aSettings );  // In inherited classes the constructor shall load up the necessary databases too. The paths will be in the setting file.

	virtual void build() = 0;
	virtual QVector< double > result() = 0;
	lpmleval::AbstractModel* model() { return mModel; }

	virtual ~AbstractOptimizer() { /*delete mModel;*/ } //It shall not delete the model, as it may come from outside.

private:

	AbstractOptimizer();

protected:

	QSettings*                mSettings;
	lpmleval::AbstractModel*  mModel;
	// In child classes there may be training and lavel databases here as well as an analytics object.
};

//-----------------------------------------------------------------------------

}
