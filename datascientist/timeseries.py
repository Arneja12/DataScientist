import matplotlib.pyplot as plt
import pandas as pd
import datascientist.plot.utils.time_series_utils as tsu
import seaborn as sns


class Plot():
    def __init__(self, df, date_column=None, series=False):
        if isinstance(df, pd.DataFrame) or isinstance(df, pd.Series):
            self.df = df
            self.series = series
            if series is False:
                x = tsu.valid_date(df[date_column].iloc[0])
                if x is not None:
                    self.date_column = date_column
        else:
            raise ValueError("Passed argument is not a DataFrame.")

    def line(self, x=None, y=None, n_rows=None, figsize=(8, 6), title=None,
             label=None, xlabel=None, ylabel=None, grid=False, style='b-',
             legend=False, legend_loc='best'):
        if self.series is False:
            if x is None or y is None:
                x, y = tsu.column_determine(self.df, x, y, self.date_column)
            plt.figure(figsize=figsize)
            if legend is True:
                if label is None:
                    label = y
            else:
                pass
            if n_rows is None:
                plt.plot(self.df[x], self.df[y], style, label=label)
            else:
                df1 = self.df.head(n_rows)
                plt.plot(df1[x], df1[y], style, label=label)
            plt.title(title)
            if legend is True:
                plt.legend(loc=legend_loc)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(grid)
            plt.show()
        else:
            if legend is True:
                if label is None:
                    label = 'Time Series'
            self.df.plot(style=style, label=label, figsize=figsize)
            if legend is True:
                plt.legend(loc=legend_loc)
            plt.grid(grid)
            plt.show()

    def subplot(self, x=None, y=None, row=1, col=None, figsize=(8, 6),
                title=None, label=None, xlabel=None, ylabel=None,
                grid=False, legend=False, legend_loc='best'):
        if self.series is False:
            color = ['red', 'green', 'orange', 'blue', 'cyan', ' magenta',
                     'yellow', 'black', 'red', 'green', 'orange', 'blue',
                     'cyan', ' magenta', 'yellow', 'black']
            if title is None:
                title = 'Time Series Plot'
            if x is None or y is None:
                x, y = tsu.column_determine(self.df, x, y, self.date_column)
            if legend is True:
                if label is None:
                    label = 'Time Series'
            if col is None:
                if row == 1:
                    fig, ax = plt.subplots(figsize=figsize)
                    ax.plot(self.df[x], self.df[y], label=y)
                    ax.set_title(title)
                else:
                    if row < 5:
                        fig, axs = plt.subplots(row, figsize=figsize)
                        fig.suptitle(title)
                        for i in range(row):
                            axs[i].plot(self.df[x], self.df[y[i]],
                                        'tab:'+color[i], label=y[i])
                    else:
                        raise ValueError('Can have only 4 plots.')
            else:
                if row > 4 or col > 4:
                    raise ValueError('Caanot have more than 4 rows or columns')
                else:
                    fig, axs = plt.subplots(row, col, figsize=figsize)
                    fig.suptitle(title)
                    if row == 1 or col == 1:
                        axs = axs.reshape(row, col)
                    k = 0
                    for i in range(row):
                        for j in range(col):
                            axs[i, j].plot(self.df[x], self.df[y[i+j+k]],
                                           label=y[i+j+k])
                        k = j
                '''Hide x labels and tick labels for top plots
                and y ticks for right plots.'''
                for ax in axs.flat:
                    ax.label_outer()
            if legend is True:
                plt.legend(loc=legend_loc)
            plt.show()
        else:
            if legend is True:
                if label is None:
                    label = 'Time Series'
            self.df.plot(subplots=True, figsize=figsize)
            if legend is True:
                plt.legend(loc=legend_loc)

    def hist(self, column=None, bins=10, stacked=False,
             orientation='vertical', cumulative=False,
             diff=False, color='b', figsize=(8, 6)):
        if column is None:
            if diff is False:
                self.df.plot(kind='hist', bins=bins, stacked=stacked,
                             orientation=orientation, cumulative=cumulative,
                             color=color, figsize=figsize)
            else:
                self.df.diff().hist(bins=bins, stacked=stacked,
                                    orientation=orientation, color=color,
                                    figsize=figsize)
        else:
            if diff is False:
                self.df[column].plot(kind='hist', bins=bins, stacked=stacked,
                                     orientation=orientation, color=color,
                                     cumulative=cumulative, figsize=figsize)
            else:
                self.df[column].diff().hist(bins=bins, stacked=stacked,
                                            orientation=orientation,
                                            color=color, figsize=figsize)
        plt.show()

    def box(self, sym='r+', box_color='DarkGreen', whisker_color='DarkOrange',
            cap_color='Gray', median_color='DarkBlue', figsize=(8, 6),
            vertical=True, positions=None, grid=False):
        colors = {'boxes': box_color, 'whiskers': whisker_color,
                  'medians': median_color, 'caps': cap_color}
        if grid is False:
            self.df.plot.box(sym=sym, color=colors, figsize=figsize,
                             vert=vertical, positions=positions)
            plt.show()
        else:
            self.df.boxplot(sym=sym, figsize=figsize, vert=vertical,
                            positions=positions)
            plt.show()

    def distplot(self, a=None, bins=None, hist=True, kde=True, rug=False,
                 fit=None, color=None, vertical=False, norm_hist=False,
                 axlabel=None, label=None, ax=None, figsize=(8, 6)):
        if a is None:
            a = tsu.single_function(self.df)
        sns.set(rc={"figure.figsize": figsize})
        sns.distplot(a=self.df[a], bins=bins, hist=hist, kde=kde, rug=rug,
                     fit=fit, color=color, vertical=vertical,
                     norm_hist=norm_hist, axlabel=axlabel, label=label, ax=ax)
        plt.plot()

    def heatmap(self, data=None, vmin=None, vmax=None, cmap=None, center=None,
                robust=False, annot=None, fmt='.2g', annot_kws=None,
                linewidths=0, linecolor='white', cbar=True, cbar_kws=None,
                cbar_ax=None, xticklabels='auto', yticklabels='auto',
                mask=None, figsize=(8, 6)):
        if data is None:
            data = tsu.Array2D(self.df, self.date_column)
        sns.set(rc={"figure.figsize": figsize})
        sns.heatmap(data=data, vmin=vmin, vmax=vmax, cmap=cmap, center=center,
                    robust=robust, annot=annot, fmt=fmt, annot_kws=annot_kws,
                    linewidths=linewidths, linecolor=linecolor, cbar=cbar,
                    cbar_kws=cbar_kws, cbar_ax=cbar_ax,
                    xticklabels=xticklabels, yticklabels=yticklabels,
                    mask=mask)
        plt.plot()

    def lag_plot(self, lag=1):
        if self.series is True:
            pd.plotting.lag_plot(self.df, lag=lag)
            plt.plot()
        else:
            x = tsu.single_function(self.df)
            pd.plotting.lag_plot(x, lag=lag)
            plt.plot()

    def autocorrelation_plot(self, ax=None):
        if self.series is True:
            pd.plotting.autocorrelation_plot(self.df, ax=ax)
            plt.plot()
        else:
            x = tsu.single_function(self.df)
            pd.plotting.autocorrelation_plot(x, ax=ax)
            plt.plot()

def _plots():
    autocorrelation_plot()
    lag_plot()
    heatmap()
    distplot()
    box()
    hist()
    subplot()
    line()