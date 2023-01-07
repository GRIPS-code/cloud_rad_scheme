%%
function [ext,ssa,asy]=read_from_pade(file,Re,shape)
        band(1,:)  =ncread(file,'Band_limits_lwr');
        band(2,:)  =ncread(file,'Band_limits_upr');
        r_lwr      = ncread(file,'Effective_Radius_limits_lwr');
        r_upr      = ncread(file,'Effective_Radius_limits_upr');
        r_ref      = ncread(file,'Effective_Radius_Ref');

        for iRe = 1:size(r_lwr,1)
            if r_lwr(iRe) <= Re & r_upr(iRe) >= Re
                break;
            end
        end

        pade_ext_p = ncread(file,'Pade_ext_p');
        pade_ext_q = ncread(file,'Pade_ext_q');
        pade_ssa_p = ncread(file,'Pade_ssa_p');
        pade_ssa_q = ncread(file,'Pade_ssa_q');
        pade_asy_p = ncread(file,'Pade_asy_p');
        pade_asy_q = ncread(file,'Pade_asy_q');

        if nargin==2
            ext = compute_from_pade(iRe, Re-r_ref(iRe),squeeze(pade_ext_p(:,:,:)),squeeze(pade_ext_q(:,:,:)));
            ssa = compute_from_pade(iRe, Re-r_ref(iRe),squeeze(pade_ssa_p(:,:,:)),squeeze(pade_ssa_q(:,:,:)));
            asy = compute_from_pade(iRe, Re-r_ref(iRe),squeeze(pade_asy_p(:,:,:)),squeeze(pade_asy_q(:,:,:)));
        else
            pade_ext_gamma_p=ncread(file,'Pade_ext_PSDshape_p');
            pade_ext_gamma_q=ncread(file,'Pade_ext_PSDshape_q');
            pade_ssa_gamma_p=ncread(file,'Pade_ssa_PSDshape_p');
            pade_ssa_gamma_q=ncread(file,'Pade_ssa_PSDshape_q');
            pade_asy_gamma_p=ncread(file,'Pade_asy_PSDshape_p');
            pade_asy_gamma_q=ncread(file,'Pade_asy_PSDshape_q');

            pade_ext_gamma_a=ncread(file,'Pade_ext_PSDshape_a');
            pade_ext_gamma_b=ncread(file,'Pade_ext_PSDshape_b');
            pade_ssa_gamma_a=ncread(file,'Pade_ssa_PSDshape_a');
            pade_ssa_gamma_b=ncread(file,'Pade_ssa_PSDshape_b');
            pade_asy_gamma_a=ncread(file,'Pade_asy_PSDshape_a');
            pade_asy_gamma_b=ncread(file,'Pade_asy_PSDshape_b');

            shape_ref = ncread(file,'Gamma_Shape_Parameter_Ref');

            ext = compute_from_pade_2d(iRe, Re-r_ref(iRe),squeeze(pade_ext_p(:,:,:)),squeeze(pade_ext_q(:,:,:)),...
                    shape,shape_ref,squeeze(pade_ext_gamma_p(:,:,:)),squeeze(pade_ext_gamma_q(:,:,:)),squeeze(pade_ext_gamma_a(:,:,:)),squeeze(pade_ext_gamma_b(:,:,:)));
            ssa = compute_from_pade_2d(iRe, Re-r_ref(iRe),squeeze(pade_ssa_p(:,:,:)),squeeze(pade_ssa_q(:,:,:)),...
                    shape,shape_ref,squeeze(pade_ssa_gamma_p(:,:,:)),squeeze(pade_ssa_gamma_q(:,:,:)),squeeze(pade_ssa_gamma_a(:,:,:)),squeeze(pade_ssa_gamma_b(:,:,:)));
            asy = compute_from_pade_2d(iRe, Re-r_ref(iRe),squeeze(pade_asy_p(:,:,:)),squeeze(pade_asy_q(:,:,:)),...
                    shape,shape_ref,squeeze(pade_asy_gamma_p(:,:,:)),squeeze(pade_asy_gamma_q(:,:,:)),squeeze(pade_asy_gamma_a(:,:,:)),squeeze(pade_asy_gamma_b(:,:,:)));
        end

end

function [y] = compute_from_pade(irad,re,p,q)
    m = size(q,3);
    n = size(p,3);
    for iband = 1:size(q,1)
        denom(iband) = q(iband,irad,1);
        for i = 2: n
            denom(iband) = q(iband,irad,i)+re*denom(iband);
        end 
        numer(iband) = p(iband,irad,1);
        for i = 2: m
            numer(iband) = p(iband,irad,i)+re*numer(iband);
        end 
    end
    y = numer./denom;
end

function [y] = compute_from_pade_2d(x,x_ref,p_size,q_size,shape,shape_ref,p_shape,q_shape,a,b)
    dx = x-x_ref;
    dshape = shape-shape_ref;
    for i=1:size(p_size,1)
        y(i) = polyval(p_size(i,:),dx) ./ polyval(q_size(i,:),dx) + ...
               polyval(p_shape(i,:),dshape)./polyval(q_shape(i,:),dshape) .* ...
               polyval(a(i,:),dx) ./ polyval(b(i,:),dx);
    end

end