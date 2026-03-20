


plot_raster_map <- function (z,x_range=NULL,y_range=NULL,
	limits,cols,coastline=NULL,coast.lwd, coast.col = "black",
	add_legend=TRUE,legend_type='add_raster_legend2',legend.pos,
	add=FALSE,
	projection=NULL,orientation=c(0,0,0),prj_parameters = NULL,
        spt.cex=1,
	missCol = "#DDDDDD", missVal = -999,
	e=NULL,limits_error=c(50,100)/100,regions='.',invert_e=TRUE,e_polygon=TRUE,
	fill.ocean=FALSE,map_db="world",ePatternRes=1, ePatternThick = 2,
			      ePointPattern = c(20,16,19,16,5,9,11), eThick = c(0.5, 0.8, 1., 1.3, 1.3, 1, 1), 
				  preLeveled = FALSE, aggregateFun = mean, 
	contour = FALSE, axes = FALSE, xaxt = 'n', yaxt ='n',
	xlab = '', ylab = '', readyCut = FALSE, interior = TRUE,...) {

    if (!is.null(limits) && length(cols)!=(length(limits)+1))
		cols=make_col_vector(cols,ncols=length(limits)+1)


    if (is.null(e)) limits_error=NULL

    if (is.null(x_range)) x_range=as.vector(extent(z))[1:2]
    if (is.null(y_range)) y_range=as.vector(extent(z))[3:4]

    z = z0 =crop(z,extent(x_range, y_range))

    if (is.null(projection)) {       
	if (!is.null(limits) & !readyCut) z=cut_results(z,limits)
	image(z,col=cols[min.raster(z,na.rm=TRUE):max.raster(z,na.rm=TRUE)], main = ' ',
	      add=add, axes = axes, xlab = xlab, ylab = ylab, xaxt = xaxt, yaxt = yaxt)
		
		#browser()
	if (!is.null(e))
            add_e(e,limits_error,cols_e,invert_e,polygons=e_polygon,ePatternRes = ePatternRes, ePatternThick = ePatternThick,
			      ePointPattern = ePointPattern, eThick = eThick, 
				  preLeveled = preLeveled, aggregateFun = aggregateFun)
		
        if (!is.null(missCol)) imageNaNs(z0,missCol,missVal)
    } else {
	c(x,y,z):=project_raster(z,projection,orientation, prj_parameters)
	z=cut_results(z,limits)
	points(x,y,col=cols[z],cex=spt.cex*0.25,pch=15)
        
        if (!is.null(e)) {
            e = raster::aggregate(e, fact = 5, method = "bilinear")
            c(x,y,e):=project_raster(e,projection,orientation, prj_parameters)
            e = 1+ max(e) - e
            for (i in 1:max(e)) {
                test = e >= i
                points(x[test], y[test],
                    col = make.transparent('black', 0.6),
                    pch = 19, cex = i*0.6*(1 + 0.33 *spt.cex))
            }
        }
    }
    if (contour) contour(z, nlevels = length(limits), lwd = par("lwd") / 2,
			 drawlabels = FALSE, add = TRUE)
    
    if (!is.null(coast.lwd))
	add_coastline(coastline,projection,orientation, prj_parameters,x_range,y_range,
		      regions = regions, fill.ocean, z,map_db = map_db,
                      interior = interior, lwd.coast = coast.lwd, col.coast = coast.col)

    if (add_legend && !is.null(limits)) match.fun(legend_type)(cols,limits,x=legend.pos,...)
}

imageNaNs <- function (z,nanCol,missVal) {
    if (is.na(missVal)) test = is.na(z) else test = z == missVal
    z[test]  = 1
    z[!test] = 0
    image(z, col = c('transparent',nanCol), add = TRUE)
}

add_e <- function(e,limits_error,cols_e,invert_e=TRUE,polygons=TRUE,ePatternThick = 2, ePatternRes=0.7, 
				  ePointPattern = c(20,16,19,16,5,9,11), eThick = c(0.5, 0.8, 1., 1.3, 1.3, 1, 1), 
				  preLeveled = FALSE, aggregateFun = mean, ...) {
	
	#if (polygons) {
	e0 = e
        factor = 5 / ePatternRes
        if (factor > 1)  e = disaggregate(e, round(factor), method = "bilinear")
            else e = aggregate(e, round(1/factor), fun = aggregateFun)
	#} else e=aggregate(e,fact=4, expand=TRUE)
	if (preLeveled) {
		e = round(e)
	} else {
		cols_e=make.transparent("grey",1-1/(length(limits_error)+1))
		e=cut_results(e,limits_error)
		if (invert_e) e=invert.raster(e,vals=1:(length(limits_error)+1))
	}
	
	add_transp <- function(lim, e, pch, cex, pattern, thick, res) {
		ee=e
		if (preLeveled) {
			ee[e == lim] = 1
			ee[e != lim] = 0
		} else {
			ee[e>lim]=1
			ee[e<=lim]=0
		}
		cells=values(ee)==1
		xy=xyFromCell(ee,which(cells))

		cols=(e[cells]-i)/(nl-1)

		cols_e=make.transparent("black",0.33)
		
		cols=make.transparent("black",0.5)
		if (polygons) {
			image(pattern.raster(ee,pattern,thick=thick,res=res),col=c('transparent',cols_e),add=TRUE)
		} else {
            points(xy,pch=pch,col=cols,cex=0.7*cex*2, bg = cols)
        }
	}
	if (preLeveled) nl = max.raster(e, na.rm = TRUE)
		else nl = length(limits_error)
		
	for (i in 1:nl) add_transp(i,e,
		ePointPattern[i],ePatternThick * eThick[i],
		c("Circle","Circle","forward-diagonal","backward-diagonal","horizontal","vertical")[i],
		c(0.1,0.3,0.6,0.5,0.5,0.5)[i]*ePatternThick,c(16,8,4,4,4,4)[i]*ePatternRes)
    
}

add_coastline <- function(coastline=NULL,projection=NULL,orientation=NULL, prj_parameters = NULL, 
                          lwd.coast=1, col.coast = "black", x_range=c(-180,180),y_range=c(-90,90),
			  fill.ocean=FALSE,rast,ocean_col="white",map_db='world',
			  interior = FALSE, ...) {
    
    if (is.null(coastline)) {
       if (fill.ocean) {
	    ## Just for Aus at the moment
	    outline <- map(map_db, regions="Australia", exact=TRUE, plot=FALSE)
	    xrange <- range(outline$x, na.rm=TRUE) # get bounding box
	    yrange <- range(outline$y, na.rm=TRUE)
	    xbox <- xrange + c(-1, 1)
	    ybox <- yrange + c(-1, 1)
	    subset <- !is.na(outline$x)
	    polypath(c(outline$x[subset], NA, c(xbox, rev(xbox))),
		     c(outline$y[subset], NA, rep(ybox, each=2)),
		     col="white",border="white", rule="evenodd")
	}
		
    addCoast <- function(lwd, interior) {
        
        if (is.null(projection)) {
	    try(map(map_db,add=TRUE,interior=interior,
                        xlim=x_range,ylim=y_range, lwd = lwd, col = col.coast,...))
	} else {
	    success = try(map(map_db,add=TRUE,interior=interior,
                    projection=projection,orientation=orientation, parameters = prj_parameters,  
		    xlim=x_range,ylim=y_range, lwd = lwd, col = col.coast,...))
        
            
	}
    }
    
    addMap(map_db, projection, prj_parameters, x_range,y_range,
		   orientation = orientation, add = TRUE, lwd = lwd.coast, col = col.coast)
    #addCoast(lwd.coast * 0.67, interior)
    #addCoast(lwd.coast, FALSE)

    } else {
	try(add_coast(coastline,lwd=lwd.coast, col = col.coast), silent = TRUE)
    }
}

project_roboinson <- function(xy) {
    xy = data.frame(xy)
    coordinates(xy) <- c("x", "y")
    proj4string(xy) <- CRS("+proj=longlat +ellps=clrk66")
    xy_p = spTransform(xy, CRS("+proj=robin +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"))
    xy_p = coordinates(xy_p)
    return(xy_p)
}

project_raster <- function(z,projection='',orientation=c(0,0,0),
                           prj_parameters = NULL, xlength=2000,ylength=1000) {

	xy=xyFromCell(z,1:ncell(z))
	z=as.matrix(values(z))

	test=!is.na(z)
	z=z[test]
            xy = xy[test,]
        if (projection == 'robinson') {
            xy_p = project_roboinson(xy)
            x = xy_p[,1]
            y = xy_p[,2]
            #plot(range(x), range(y), type =  'n', axes = FALSE, xlab = '', ylab = '')
        } else {
	    xy=mapproject(xy[,1],xy[,2],projection=projection,orientation=orientation, parameters = prj_parameters)
	    x=xy[[1]]
	    y=xy[[2]]
        }
	xyz=cbind(x,y,z)
    
    return(list(x,y,z))
}

cut_results <- function(pdata,limits) {
	nlimits=length(limits)
	if (class(pdata)=="RasterBrick" && nlayers(pdata)==1) pdata=pdata[[1]]
	odata=pdata
	odata[]=nlimits+1
	for (i in nlimits:1) odata[pdata<=limits[i]]=i
	odata[is.na(pdata)]=NaN
	return(odata)
}

add_raster_legend <- function(cols,limits,labelss=NULL,x='left',add=TRUE,main_title='',
			      spt.cex=3,pch=15,between=FALSE,y.intersp=1,bty='n',
			      title_line=-2,title_adj=0.06,title_cex=0.8,mar=NULL,...) {

	if (length(cols)!=length(limits)+1)
		cols=make_col_vector(cols,ncols=length(limits)+1)
	mar0=par("mar")
	if (!is.null(mar)) par(mar=mar) else mar=mar0
	model_legend <- function(cols,labelss,...) legend(x=x,legend=labelss,col=cols,xpd=TRUE,...)
	testp=1
	bg.col="#000000"

	if (add==FALSE) plot.new()

	if (is.null(labelss)) {
		limits=round(limits,5)
		labelss=c(paste("<",limits[1]),
				  paste(limits[-length(limits)],"-",limits[-1]),
				  paste(">",limits[length(limits)]))
		if (limits[1]==0) labelss[1]='0'
	}

	test=(as.numeric(labelss[1])==0)
	if (length(labelss)==(length(cols)-1)) {
		labelss=c("",labelss)
	} else test=FALSE
	if (between) {
		cols=rep(cols,each=10)

		labelss=rep(labelss,each=10)
		labelss[(1:length(labelss))[-seq(6,length(labelss),10)]]=""
		if (test) {

			labelss[10]=labelss[16]
			labelss[16]=""
		}

		cols=cols[-(1:5)]
		labelss=labelss[-(1:5)]
		cols[(1:3)]="transparent"
		bg.col=rep(bg.col,length(cols))
		bg.col[(1:3)]="transparent"

		y.intersp=y.intersp/10
	}

	model_legend(rev(bg.col),rev(labelss),x,pt.cex=1.05*spt.cex,pch=pch,y.intersp=y.intersp,bty=bty,...)
	model_legend(rev(cols),rev(labelss),x,pt.cex=spt.cex,pch=pch,y.intersp=y.intersp,bty=bty,...)

	mtext(main_title,side=1,line=title_line+.16-0.6*(5-length(limits))+mar[1]*0.5,cex=title_cex,adj=title_adj)
	par(mar=mar0)
}

add_raster_legend2 <-function(cols,limits,labelss=NULL,x='bottomleft', dat = NULL,
			      add=TRUE,spt.cex=2,pch=15,main_title='',
			      plot_loc=c(0.22,0.7,0.07,0.10+0.05*(length(e_lims)-1)),
			      e_lims=0,labelss.cex=1,title_cex=0.8,srt=90,transpose=TRUE,
			      invert_e=TRUE,polygons=TRUE,elim_lab=TRUE,elim_txt=TRUE,
			      oneSideLabels = TRUE, ylabposScling=1,units = NULL, 
                              maxLab = NULL, extend_max = FALSE, extend_min = FALSE, ...) {
	#source("transpose_cor.r")
	
    if (length(cols)!=length(limits)+1)
	cols=make_col_vector(cols,ncols=length(limits)+1)

    if (!add) {
	plot.new()
	add=TRUE
    }

    if (is.null(e_lims)) {
	e_lims=0
	cols_e=make.transparent("grey",1-1/(length(e_lims)+1))
    }

    cal_frac_of_plot_covered <- function(index=1:2) {
	xlims=par("usr")[index]
	xlims[3]=xlims[2]-xlims[1]
	return(xlims)
    }

    if (add) {
	xlims=cal_frac_of_plot_covered(1:2)
	ylims=cal_frac_of_plot_covered(3:4)
    } else {
	xlims=c(0,1,1)
	ylims=c(0,1,1)
        plot(c(0,1),c(0,1))
    }
    nlims=length(limits)+1

    x=xlims[1]+plot_loc[1]*xlims[3]+(plot_loc[2]-plot_loc[1])*xlims[3]*(matrix(1:nlims,nlims,1)-0.5)/nlims

    y=seq(ylims[1]+ylims[3]*plot_loc[3],ylims[1]+ylims[3]*plot_loc[4],length.out=length(e_lims)+1)
    z=matrix(1:nlims,nlims,1+length(e_lims))
    c(xx,yy,z):=transpose_cor(transpose,x,y,z)
    image(x=xx,y=yy,z=z,col=cols[1:nlims],add=TRUE,
    	  xaxt='n',yaxt='n',xlab='',ylab='',xdp=TRUE, xpd = NA)
        

    if (length(e_lims)>1) {
        z=t(matrix(1:(1+length(e_lims)),1+length(e_lims),nlims))
    	if (invert_e) z=invert(z)
    	image_z_gt_i <- function(i,pch,cex,pattern,thick,res) {
    	    zz=z
    	    zz[z>i]=1
    	    zz[z<=i]=0

    	    c(x,y,zz):=transpose_cor(transpose,x,y,zz)
    	    xx=rep(x,length(y))
    	    yy=rep(y,each=length(x))
    	    zz=as.vector(zz)
    	    cols=c('transparent',make.transparent("black",1-0.5*i/length(e_lims)))
    	    cols_e=make.transparent("black",1-0.75*i/length(e_lims))

    	    if (polygons) {
    	        zz=rasterFromXYZ(cbind(xx,yy,zz))
    	        image(pattern.raster(disaggregate(zz,100),
                      pattern,,thick=thick,res=res*25),
                      col=c('transparent',cols_e),add=TRUE, xpd = NA)
    	    } else points(xx,yy,col=cols[zz+1],pch=pch,cex=cex*1.7, xpd = NA)
    		#image(x=x,y=y,z=zz,col=c('transparent',cols_e),add=TRUE)
    	}

    	for (i in 1:length(e_lims))
    	    image_z_gt_i(i,c(4,3,1,16,5,9,11)[i],c(1,1,1.3,1.3,1.3,1,1)[i],
			 c("Circle","Circle","forward-diagonal","backward-diagonal",
			   "horizontal","vertical")[i],
			 c(0.1,0.67,0.5,0.5,0.5,0.5)[i],c(16,8,4,4,4,4)[i])
    }
    dx=(x[2]-x[1])/2
    xp=c(min(x)-dx,min(x)-dx,max(x)+dx,max(x)+dx)

    yp=c(y[1]-diff(range(y))/(length(e_lims)*2),tail(y,1)+diff(range(y))/(length(e_lims)*2))

    c(xx,yy):=transpose_cor(transpose,xp,c(yp,rev(yp)))
    polygon(xx,yy,xpd=NA)
    xxp = xx; yyp = yy
    yl=seq(yp[1],yp[2],length.out=length(e_lims)+2)

    if (!polygons && length(e_lims)>1) {
        if (!transpose)
            lapply(yl,function(y) lines(xp[2:3],rep(y,2),lwd=0.5,
                   col=make.transparent("black",0.5), xpd = NA))
        else 
            lapply(yl,function(y) lines(rep(y,2),xp[2:3],lwd=0.5,
                   col=make.transparent("black",0.5), xpd = NA))
    }

    #===============================
    ytop=tail(yp,1)+tail(diff(y),1)*ylabposScling
    ybottom=yp[1]-diff(y[1:2]*ylabposScling)
    #if (length(e_lims)>1) ytop=ybottom

    xt = c(x[1] - dx, x[1], x + dx)
    if (is.na(oneSideLabels)) ybottom = ytop
	else if (oneSideLabels) ytop = ybottom
    yt=c(ytop,rep(c(ytop,ybottom),length.out=length(xt)-1))

    if (is.null(labelss)) {
	lab1  = ''#paste("<",limits[1],sep="")
	labN  = ''#paste(limits[length(limits)],"+")
	dlim1 = diff(limits[1:2]); dlimN = diff(rev(tail(limits,2)))

	if (!is.null(dat)) {
	    if (limits[1]>0 && limits[1]<=dlim1 && min.raster(dat,na.rm=TRUE)>=0)
                lab1 = "0"
	    else if (tail(limits,1)<0 && tail(limits,1)<=dlimN 
                   && max.rater(dat,na.rm=TRUE)<=0) labN = "0"
	}

	labelss=c(lab1,"",limits,labN)
	if (limits[1]==0) {
	    labelss[1]=""
	    labelss[2]='0'
	    labelss[3]=""
        }
    } else {
	if(class(labelss)=='list' && length(labelss)==1) labelss=labelss[[1]]

	if (limits[1]==0) {
	    labelss=c("",labelss[1],"",labelss[-1],"")
	} else {
	    labelss=c(labelss[1],"",labelss[-1],"")
	}

	if (length(labelss)==(length(xt)-2)) labelss=c("",labelss,"")
	if (length(labelss)==(length(xt)-1)) labelss=c("","",labelss[-2])
    }
        
    if (length(labelss)==0) labelss=rep("",length(xt))

	
    c(xx,yy):=transpose_cor(transpose,xt,yt)
    if (transpose) {
	    srt = srt-90
	    adj = 1
    }  else {
	adj = 0.5
    	yy  = yy - 0.1
    }

    add_pointy_end <- function(x_point) {
        y_point = yyp[1:2]
        y_mid = mean(y_point)
        ploy_line <- function(i, di) {
            polygon(c(x_point, rev(x_point)), c(i + di, y_mid + di, i + di*2,i + di*2),
                    col = "white", border = "white", lwd = 2, xpd = NA)
            lines(x_point, c(i, y_mid), xpd = NA)
        }
       mapply(ploy_line, y_point, 0.1 * diff(y_point) * c(-1, 1))
    }

    if (extend_max & !transpose) {
        x_point = tail(xx,2)
        add_pointy_end(x_point)
    }
    if (extend_min & !transpose) {
        x_point = xx[c(3,1)]
        add_pointy_end(x_point)
    }
    
    if (!is.null(maxLab)) labelss[length(labelss)] = maxLab
    if (!is.null(units)) {
        adj = rep(adj, length(labelss))
        index = labelss!="" & labelss !="0"
        if (nchar(units) > 2 && as.character(units) != "~DEG~C") {
            if (!oneSideLabels && tail(labelss, 1) != "") {
                labelss = c(labelss, "")
                xx = c(xx, tail(xx, 1))
                yy = c(yy, tail(yy,2)[1])
                adj = c(adj, 0.1)
            }
            index = length(labelss)
            #adj[index] = 0.1
        }        
        labelss[index] = paste0(labelss[index], units)       
      }
    
    mapply(text.units,x=xx+0.0667*(length(e_lims)-1),y=yy,
           labelss,xpd=NA,cex=labelss.cex,srt=srt,adj = adj)

    
	#===============================
	if (length(e_lims)!=1) {
		c(xx,yy):=transpose_cor(transpose,xt[1]-2*diff(xt[1:2]),c(y[1]-diff(y[1:2]),y)-0.33*c(diff(y[1:3]),diff(y))/2)
		srt=0
		if (transpose) {srt=srt+90; xx=xx+diff(xx[1:2])}
		if(elim_txt)
			text(x=xx,y=yy-0.05,
			 	 c(0,e_lims,''),pos=3,cex=labelss.cex*0.95,
				 xpd=NA,srt=srt)

		if (elim_lab) mtext("sd/mean (%)",side=c(2,1)[transpose*1+1],
				    cex=title_cex,line=-1+transpose*1.0,xpd=NA)
	}

	for (i in list(c(1,3),c(2))[[transpose*1+1]])
		mtext(main_title,side=i,cex=title_cex,line=-1-transpose*0.1)


}

add_coast <- function(shape_file="libs/AusCoast.csv",...) {
    hcoast=read.csv(shape_file)

    for (isle in 1:max(hcoast[,2])) {
	test=hcoast[,2]==isle
    	lines(hcoast[test,3],hcoast[test,4],...)
    }
}
