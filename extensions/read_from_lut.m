function [ext,ssa,asy]=read_from_lut(file,Re)
    band(1,:)  =ncread(file,'Band_limits_lwr');
    band(2,:)  =ncread(file,'Band_limits_upr');
    r_lwr      = ncread(file,'Effective_Radius_limits_lwr');
    r_upr      = ncread(file,'Effective_Radius_limits_upr');

    iRe = find(Re>=r_lwr & Re<r_upr);

    lut_ext_a = ncread(file,'ext_coef_a');
    lut_ext_b = ncread(file,'ext_coef_b');
    lut_ssa_a = ncread(file,'ssa_coef_a');
    lut_ssa_b = ncread(file,'ssa_coef_b');
    lut_asy_a = ncread(file,'asy_coef_a');
    lut_asy_b = ncread(file,'asy_coef_b');

    ext = compute_from_lut(Re,squeeze(lut_ext_a(:,iRe,:)),squeeze(lut_ext_b(:,iRe,:)));
    ssa = compute_from_lut(Re,squeeze(lut_ssa_a(:,iRe,:)),squeeze(lut_ssa_b(:,iRe,:)));
    asy = compute_from_lut(Re,squeeze(lut_asy_a(:,iRe,:)),squeeze(lut_asy_b(:,iRe,:)));
end

function [y] = compute_from_lut(x,a,b)
    y = a + b .* x;
end